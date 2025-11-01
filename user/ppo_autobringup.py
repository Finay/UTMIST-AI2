import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
from stable_baselines3 import PPO


mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

def check_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor, threshold: float = 0.99) -> bool:
    """
    Check if the Pearson correlation coefficient (PCC) between two tensors exceeds a given threshold.
    
    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
        threshold (float): Minimum acceptable correlation (default: 0.99).
    
    Returns:
        bool: True if PCC >= threshold, else False.
    """
    # Flatten tensors to 1D
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    # Ensure same shape
    if t1.shape != t2.shape:
        raise ValueError("Input tensors must have the same number of elements")

    # Compute Pearson correlation coefficient
    t1_mean = t1.mean()
    t2_mean = t2.mean()
    numerator = torch.sum((t1 - t1_mean) * (t2 - t2_mean))
    denominator = torch.sqrt(torch.sum((t1 - t1_mean) ** 2) * torch.sum((t2 - t2_mean) ** 2))
    pcc = numerator / denominator

    # Check if it exceeds threshold
    return pcc.item() >= threshold

# Feature Extractor
#   Check if model.policy.pi_features_extractor has model attr
# MLP Extractor Policy
#   
# Action Net

class TTPolicyNet(nn.Module):
    def __init__(self, policy_net):
        super(TTPolicyNet, self).__init__()
        self.raw_state_dict = policy_net.state_dict()
        self.n_layers = len(self.raw_state_dict) // 2 # state_dict has a weight and bias for each layer

        self.weights = []
        self.biases = []
        for index in range(0, self.n_layers * 2, 2):
            weight = ttnn.from_torch(
            self.raw_state_dict[f"{index}.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            )
            bias = ttnn.from_torch(self.raw_state_dict[f"{index}.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.weights.append(weight)
            self.biases.append(bias)


    def forward(self, tensor: ttnn.Tensor):
        for index in range(0, self.n_layers):
            prev_tensor = tensor
            tensor = ttnn.tanh(ttnn.linear(tensor, self.weights[index], bias=self.biases[index])) # Default policy_net activation is tanh
            prev_tensor.deallocate()
        return tensor


class TTActionNet(nn.Module):
    def __init__(self, action_net):
        super(TTActionNet, self).__init__()
        self.raw_state_dict = action_net.state_dict()
        assert set(self.raw_state_dict.keys()) == set(['weight', 'bias']), "Unexpected action_net state_dict"

        self.weight = ttnn.from_torch(
            self.raw_state_dict["weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.bias = ttnn.from_torch(self.raw_state_dict["bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, tensor: ttnn.Tensor):
        return ttnn.linear(tensor, self.weight, bias=self.bias)


class TTPolicy(nn.Module):
    def __init__(self, model: PPO):
        super(TTPolicy, self).__init__()
        self.base_model = model
        self.tt_policy_net = TTPolicyNet(model.policy.mlp_extractor.policy_net)
        self.tt_action_net = TTActionNet(model.policy.action_net)
        

    def forward(self, obs, run_pcc_validation=False):
        features = self.base_model.policy.pi_features_extractor(torch.from_numpy(obs).unsqueeze(0)) # Use given feature extractor
        tt_features = ttnn.from_torch(features, device=mesh_device, layout=ttnn.TILE_LAYOUT)
        action_latent = self.tt_policy_net(tt_features)
        action = self.tt_action_net(action_latent)
        torch_action = ttnn.to_torch(action).flatten().to(torch.float32)
        processed_action = torch.clip(torch_action, torch.zeros_like(torch_action), torch.ones_like(torch_action))
        
        if run_pcc_validation:
            true_action_latent = self.base_model.policy.mlp_extractor.policy_net(features)
            assert check_pcc(true_action_latent, ttnn.to_torch(action_latent).flatten().to(torch.float32)), "Failed policy_net pcc check"
            true_action = self.base_model.policy.action_net(true_action_latent)
            assert check_pcc(true_action, torch_action), "Failed action_net pcc check"
            true_full_pass, _ = self.base_model.predict(obs, deterministic=True)
            if check_pcc(torch.from_numpy(true_full_pass), processed_action):
                print("Passed full-pass pcc check")
            else:
                print("Failed full-pass pcc check")
        
        tt_features.deallocate()
        action_latent.deallocate()
        action.deallocate()

        return processed_action.numpy(), None


    def predict(self, obs, run_pcc_validation=False):
        return self.forward(obs, run_pcc_validation=run_pcc_validation)
