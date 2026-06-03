import torch
import gymnasium as gym
import numpy as np

from dqn_agent import QNetwork

NUM_EVAL_EPISODES = 10

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = QNetwork(state_dim, action_dim).to(device)
q_net.load_state_dict(torch.load("results/dqn_cartpole.pth", map_location=device))
q_net.eval()

rewards = []
for ep in range(1, NUM_EVAL_EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = q_net(state_t).argmax(dim=1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    rewards.append(total_reward)
    print(f"Episode {ep:2d}  reward: {total_reward:.1f}")

env.close()
print(f"\nMean reward over {NUM_EVAL_EPISODES} episodes: {np.mean(rewards):.1f}")
