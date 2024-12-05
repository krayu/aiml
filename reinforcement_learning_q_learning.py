import numpy as np
import random

# GridWorld environment where agent navigates a grid
class GridWorld:
    def __init__(self, size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.size, self.start, self.goal = size, start, goal
        self.pos = start
    
    # Resets the agent to the starting position
    def reset(self):
        self.pos = self.start
        return self.pos
    
    # Takes an action and returns new state, reward, and whether goal is reached
    def step(self, action):
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Directions: left, right, up, down
        move = moves[action]
        new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
        if 0 <= new_pos[0] < self.size[0] and 0 <= new_pos[1] < self.size[1]:
            self.pos = new_pos
        return self.pos, -1 if self.pos != self.goal else 0, self.pos == self.goal

# Q-learning agent that learns the best policy to navigate the grid
class QLearningAgent:
    def __init__(self, env, lr=0.1, df=0.9, er=1.0, ed=0.995):
        self.env, self.lr, self.df, self.er, self.ed = env, lr, df, er, ed
        self.q = np.zeros((env.size[0], env.size[1], 4))  # Q-table for each position and action
    
    # Chooses an action based on exploration (random) or exploitation (best known action)
    def choose_action(self, state):
        if random.random() < self.er:  # Exploration: random action
            return random.choice([0, 1, 2, 3])  # Up, Down, Left, Right
        return np.argmax(self.q[state])  # Exploitation: best action based on Q-values
    
    # Updates the Q-table based on the received reward and the next state
    def learn(self, state, action, reward, next_state, done):
        old_q = self.q[state][action]  # Old Q-value
        next_max_q = np.max(self.q[next_state])  # Max Q-value for next state
        # Q-learning formula to update Q-value
        self.q[state][action] = old_q + self.lr * (reward + self.df * next_max_q - old_q)
        if done: self.er *= self.ed  # Decay exploration rate after each episode

# Initialize environment and agent
env = GridWorld()
agent = QLearningAgent(env)

# Training loop: agent learns over 1000 episodes
for _ in range(1000):
    state, done = env.reset(), False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state

# Test learned policy: agent follows learned strategy
state, done = env.reset(), False
while not done:
    action = agent.choose_action(state)
    state, _, done = env.step(action)
    print(state)