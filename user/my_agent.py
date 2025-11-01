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

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


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

        # To run a TTNN model, you must maintain a pointer to the device and can be done by 
        # uncommmenting the line below to use the device pointer
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

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

        # To run the sample TTNN model during inference, you can uncomment the 5 lines below:
        # This assumes that your self.model.policy has the MLPPolicy architecture defined in `train_agent.py` or `my_agent_tt.py`
        # mlp_state_dict = self.model.policy.features_extractor.model.state_dict()
        # self.tt_model = TTMLPPolicy(mlp_state_dict, self.mesh_device)
        # self.model.policy.features_extractor.model = self.tt_model
        # self.model.policy.vf_features_extractor.model = self.tt_model
        # self.model.policy.pi_features_extractor.model = self.tt_model

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1U2ZehCJBrmcTvaiu5eoEWUPxwaczthf_/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
