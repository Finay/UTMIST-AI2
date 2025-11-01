import os
print(os.getcwd())
import sys
sys.path.insert(1, os.getcwd())

from environment.environment import WarehouseBrawl
from user.my_agent import SubmittedAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn

env = WarehouseBrawl()
my_agent = SubmittedAgent()
my_agent.get_env_info(env)

print(my_agent.model.policy)
