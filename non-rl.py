import gymnasium as gym
import gymnasium_environments
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

plt.ion()

# --- Configuration ---
EPISODES = 10
MAX_STEPS_PER_EPISODE = 100
RUN_NAME = "Non RL Graph (Rule Based Agent)"

# Setup
env = gym.make("gymnasium_environments/CreateRedBall-v0", render_mode="human")
os.makedirs("./logs", exist_ok=True)
os.makedirs("./screenshots", exist_ok=True)
log_file = f"./logs/{RUN_NAME}.txt"
plot_file = f"./screenshots/{RUN_NAME}.png"

episode_rewards = []
episode_steps = []

fig, ax = plt.subplots(figsize=(12, 6))
line1, = ax.plot([], [], label='Episode Reward', color='green')
line2, = ax.plot([], [], label='Episode Steps', color='blue')
ax.set_title(f"{RUN_NAME}")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward / Steps")
ax.grid(True)
ax.legend()
plt.tight_layout()

def choose_action(observation):
    if observation < 320:
        return min(observation + 20, 640)
    elif observation > 320:
        return max(observation - 20, 0)
    else:
        return 320

def log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    print(msg)

# Main loop
try:
    with open(log_file, "w") as f:
        f.write(f"Rule-based agent run: {RUN_NAME}\n\n")

    for episode in range(EPISODES):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        log(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")

        # Update plot
        line1.set_data(range(1, len(episode_rewards)+1), episode_rewards)
        line2.set_data(range(1, len(episode_steps)+1), episode_steps)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

except KeyboardInterrupt:
    log("⚠️ Interrupted by user.")
except Exception as e:
    log(f"❌ Exception occurred: {e}")
finally:
    plt.ioff()
    fig.savefig(plot_file)
    log(f"✅ Saved plot to {plot_file}")
    env.close()
    plt.close('all')
