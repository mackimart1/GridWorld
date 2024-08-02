import numpy as np
import random

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.agent_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 3)]
        
        for obs in self.obstacles:
            self.grid[obs] = -1
        self.grid[self.goal_pos] = 1

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = tuple(np.add(self.agent_pos, directions[action]))
        
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos

        reward = -1  # Small negative reward for each step
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        elif self.grid[self.agent_pos] == -1:
            reward = -10
            done = True

        return self.agent_pos, reward, done

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

def train(episodes=1000):
    env = GridWorld()
    agent = QLearningAgent(env.size, 4)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    return agent

if __name__ == "__main__":
    trained_agent = train()
    print("Training completed!")