*Grid World Q-Learning Agent*

Overview
This repository implements a Q-Learning agent to solve a Grid World problem. The agent learns to navigate a 5x5 grid and reach the goal position while avoiding obstacles.
Dependencies
NumPy (import numpy as np)
Random (import random)
Classes
GridWorld

Represents the Grid World environment.

__init__(size=5): Initializes the grid with the specified size, agent position, goal position, and obstacles.
reset(): Resets the agent position to the starting position.
step(action): Updates the agent position based on the action taken and returns the new position, reward, and done status.
QLearningAgent
Represents the Q-Learning agent.
__init__(state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1): Initializes the Q-table, learning rate, discount factor, and epsilon.
get_action(state): Returns an action based on the epsilon-greedy policy.
update(state, action, reward, next_state): Updates the Q-table based on the Q-Learning update rule.
Functions
train(episodes=1000)
Trains the Q-Learning agent for the specified number of episodes.
Usage
To train the agent, simply run the script. The agent will be trained for 1000 episodes by default.


Python

if __name__ == "__main__":
    trained_agent = train()
    print("Training completed!")
Notes
The grid size, learning rate, discount factor, and epsilon can be adjusted for different scenarios.
The Q-table is initialized with zeros, but other initialization methods can be explored.
The agent's performance can be improved by tuning the hyperparameters or using more advanced exploration strategies.