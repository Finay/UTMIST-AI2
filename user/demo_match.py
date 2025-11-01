import os
import sys
sys.path.insert(1, os.getcwd())

from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

my_agent = SubmittedAgent()

#NOTE: Input your file path here in SubmittedAgent if you are loading a model:
opponent = BasedAgent()

match_time = 90

# Run a single real-time match
result = run_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
if f"{result.player1_result}" == "Result.WIN":
    run_match(
        agent_1=my_agent,
        agent_2=opponent,
        max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
        resolution=CameraResolution.LOW,
        video_path='tt_agent.mp4' 
    )
else:
    print(f"Player lost test match {result.player1_result}")
