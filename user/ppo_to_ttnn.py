import os
print(os.getcwd())
import sys
sys.path.insert(1, os.getcwd())

from environment.environment import WarehouseBrawl
from user.my_agent import SubmittedAgent, _process_obs

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import ttnn
except:
    print("No TTNN available")

env = WarehouseBrawl()

my_agent = SubmittedAgent()
my_agent.get_env_info(env)

tt_mod_agent = SubmittedAgent()
tt_mod_agent.get_env_info(env)


for key, value in my_agent.model.policy.state_dict().items():
    print(f'{key}: {value.shape}')


class TTMLPExtractorPolicy(nn.Module):
    def __init__(self, state_dict, mesh_device):
        super(TTMLPExtractorPolicy, self).__init__()
        self.mesh_device = mesh_device

        self.fc1 = ttnn.from_torch(
            state_dict["mlp_extractor.policy_net.0.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc2 = ttnn.from_torch(
            state_dict["mlp_extractor.policy_net.2.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc3 = ttnn.from_torch(
            state_dict["mlp_extractor.policy_net.4.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.fc1_b = ttnn.from_torch(state_dict["mlp_extractor.policy_net.0.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc2_b = ttnn.from_torch(state_dict["mlp_extractor.policy_net.2.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc3_b = ttnn.from_torch(state_dict["mlp_extractor.policy_net.4.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        print("Running TTMEP forward pass!")
        obs = obs.to(torch.bfloat16)
        tt_obs = ttnn.from_torch(obs, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

        x1 = ttnn.tanh(ttnn.linear(tt_obs, self.fc1, bias=self.fc1_b))
        tt_obs.deallocate()

        x2 = ttnn.tanh(ttnn.linear(x1, self.fc2, bias=self.fc2_b))
        x1.deallocate()

        x3 = ttnn.tanh(ttnn.linear(x2, self.fc3, bias=self.fc3_b))
        x2.deallocate()

        tt_out = ttnn.to_torch(x3).flatten().to(torch.float32)
        x3.deallocate()

        return tt_out


class TTActionNet(nn.Module):
    def __init__(self, state_dict, mesh_device):
        super(TTActionNet, self).__init__()
        self.mesh_device = mesh_device

        self.fc1 = ttnn.from_torch(
            state_dict["action_net.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.fc1_b = ttnn.from_torch(state_dict["action_net.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        print("Running TTAN forward pass!")
        obs = obs.to(torch.bfloat16)
        tt_obs = ttnn.from_torch(obs, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

        x = ttnn.linear(tt_obs, self.fc1, bias=self.fc1_b)
        tt_obs.deallocate()

        tt_out = ttnn.to_torch(x).flatten().to(torch.float32)
        x.deallocate()

        return tt_out


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


def test_mlp_policy():
    
    # Open the device (since we are only using single devices N150 cards, your mesh shape will be 1x1)

    # Create torch input
    obs, info = env.reset()
    x = torch.tensor(obs[0]).unsqueeze(0)

    # Create TTNN model, we pass the torch model state dict to use its weights
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))
    tt_mep = TTMLPExtractorPolicy(my_agent.model.policy.state_dict(), mesh_device)
    tt_an = TTActionNet(my_agent.model.policy.state_dict(), mesh_device)

    x1 = torch.randn(1, 86).to(torch.float)
    print(f'MEP pcc check: {check_pcc(my_agent.model.policy.mlp_extractor.policy_net(x1), tt_mep(x1))}')
    x2 = torch.randn(1, 64).to(torch.float)
    print(f'AN pcc check: {check_pcc(my_agent.model.policy.action_net(x2), tt_an(x2))}')
    

    tt_mod_agent.model.policy.mlp_extractor.policy_net = tt_mep
    tt_mod_agent.model.policy.action_net = tt_an

    features = _process_obs(x.squeeze()).unsqueeze(0)
    with ttnn.tracer.trace():
        l = tt_an(tt_mep(features))
        true_tt_y = torch.clip(l, torch.zeros_like(l), torch.ones_like(l))

    # Run forward pass
    y = torch.from_numpy(my_agent.predict(x))
    tt_y = torch.from_numpy(tt_mod_agent.predict(x))

    # Check that the Pearson Correlation Coefficient is above 0.99 (meaning that these 2 tensors are very very close to eachother) to check for correctness
    if check_pcc(y, tt_y):
        print("✅ PCC check passed!")
    else:
        print("❌ PCC below threshold.")
        print(tt_y)
        print(y)

    if check_pcc(y, true_tt_y):
        print("✅ PCC check passed!")
    else:
        print("❌ PCC below threshold.")
        print(true_tt_y)
        print(y)

    ttnn.tracer.visualize(true_tt_y, file_name="net_trace.svg")

if __name__ == "__main__":
    test_mlp_policy()


# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L636