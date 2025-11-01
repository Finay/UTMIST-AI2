# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import torch
from torch.nn import functional as F

import ttnn
import torch.nn as nn
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



def _process_obs(obs):
    offset = 32
    return torch.cat((
        obs[:8],
        F.one_hot(obs[8].to(torch.int64), num_classes=13),
        obs[9:13],
        F.one_hot(obs[14].to(torch.int64), num_classes=13),
        F.one_hot(obs[15].to(torch.int64), num_classes=3),
        obs[offset:offset+8],
        F.one_hot(obs[offset+8].to(torch.int64), num_classes=13),
        obs[offset+9:offset+13],
        F.one_hot(obs[offset+14].to(torch.int64), num_classes=13),
        F.one_hot(obs[offset+15].to(torch.int64), num_classes=3),
        obs[28:32]
    ))


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomExtractor, self).__init__(observation_space, 86)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_processed = torch.stack([_process_obs(obs[i, :]) for i in range(obs.shape[0])]).to(torch.float32)
        return obs_processed

    @classmethod
    def get_policy_kwargs(cls, hidden_dim: int = 64, hidden_layers: int = 3) -> dict:
        return dict(
            features_extractor_class=cls,
            net_arch=[hidden_dim]*hidden_layers
        )


class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)


    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                policy_kwargs=CustomExtractor.get_policy_kwargs(),
                n_steps=30 * 90 * 5,
                batch_size=30 * 5,
                ent_coef=0.01,
            )
            del self.env
        else:
            self.model = PPO.load(self.file_path, custom_objects={
                'policy_kwargs': CustomExtractor.get_policy_kwargs(),
            })
        self.tt_policy = TTPolicy(self.model)


    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1U2ZehCJBrmcTvaiu5eoEWUPxwaczthf_/view?usp=sharing"
            try:
                gdown.download(url, output=data_path, fuzzy=True)
            except:
                from urllib.request import urlretrieve
                urlretrieve("http://20.109.2.14:8080/experiments/heater2.1/rl_model_62967869_steps.zip", data_path)
        return data_path


    def predict(self, obs):
        action, _ = self.tt_policy.predict(obs, run_pcc_validation=True)
        return action
