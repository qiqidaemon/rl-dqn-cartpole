import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from dqn_agent import DQNAgent

# ── Hyperparameters ──────────────────────────────────────────────────────────
NUM_EPISODES = 1300
MAX_STEPS = 500
SOLVED_REWARD = 475   # CartPole-v1 is "solved" at avg 475 over 100 episodes
PRINT_EVERY = 10

# ── Setup ────────────────────────────────────────────────────────────────────
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n              # 2

agent = DQNAgent(state_dim, action_dim)

os.makedirs("results", exist_ok=True)

# ── Training loop ─────────────────────────────────────────────────────────────
episode_rewards = []
running_avg = []

print(f"Training DQN on CartPole-v1  |  device: {agent.device}")
print("─" * 55)

for episode in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action, reward, next_state, done)
        agent.learn()

        state = next_state
        total_reward += reward

        if done:
            break

    agent.decay_epsilon()
    agent.update_target(episode)
    episode_rewards.append(total_reward)

    avg = np.mean(episode_rewards[-100:])
    running_avg.append(avg)

    if episode % PRINT_EVERY == 0:
        print(f"Episode {episode:4d}  reward: {total_reward:6.1f}  "
              f"avg(100): {avg:6.1f}  eps: {agent.epsilon:.3f}")

    if avg >= SOLVED_REWARD and episode >= 100:
        print(f"\nSolved at episode {episode}!  avg(100) = {avg:.1f}")
        break

env.close()

# ── Save model ────────────────────────────────────────────────────────────────
import torch
torch.save(agent.q_net.state_dict(), "results/dqn_cartpole.pth")
print("Model saved → results/dqn_cartpole.pth")

# ── Plot training curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(episode_rewards, alpha=0.4, color="steelblue", label="Episode reward")
ax.plot(running_avg, color="steelblue", linewidth=2, label="Running avg (100 ep)")
ax.axhline(SOLVED_REWARD, color="green", linestyle="--", linewidth=1.5, label=f"Solved threshold ({SOLVED_REWARD})")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("DQN on CartPole-v1 — Training Curve")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_curve.png", dpi=150)
print("Training curve saved → results/training_curve.png")
