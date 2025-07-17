import gymnasium
import blocksworld_env
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

def train_qlearning(env, episodes, gamma, epsilon, epsilon_min, decay, alpha, run_name="default"):
    # Initialize Q-table
    numstates = env.observation_space.n
    numactions = env.action_space.n
    qtable = np.random.rand(numstates, numactions).tolist()

    # Prepare plotting
    steps_per_episode = []
    rewards_per_episode = []
    fig, ax = plt.subplots(figsize=(16, 9))
    line, = ax.plot([], [], label='Steps per Episode', color='Blue')
    line2, = ax.plot([], [], label='Rewards per Episode', color='Green')
    ax.set_xlim(0, episodes)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps / Cumulative Rewards')

    hyperparams_str = f"Gamma: {gamma}, Epsilon: {epsilon}, Decay: {decay}, Alpha: {alpha}"
    ax.set_title(f'Q-Learning on Blocks World for {run_name} using {env.spec.id}', fontsize=12)
    ax.grid(True)
    plt.legend()

    # Prepare log file and clear previous content
    os.makedirs("./logs", exist_ok=True)
    log_filename = f"./logs/training_log_{run_name}.txt"
    with open(log_filename, "w") as f:
        f.write("Training Log\n")
        f.write(f"Hyperparameters: {hyperparams_str}\n\n")

    for i in range(episodes):
        state, info = env.reset()
        steps = 0
        total_reward = 0
        done = False

        while not done:
            os.system('clear')
            print(f"Episode {i+1} / {episodes}")
            if env.render_mode == "human":
                env.render()

            steps += 1

            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = qtable[state].index(max(qtable[state]))

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            qtable[state][action] = (1 - alpha) * qtable[state][action] + alpha * (
                reward + gamma * max(qtable[next_state])
            )

            state = next_state

        # Log episode result
        with open(log_filename, "a") as f:
            f.write(f"Episode {i+1}: Steps {steps}, Total Reward {total_reward}\n")

        # Decay epsilon exponentially
        epsilon = max(epsilon_min, epsilon - decay * epsilon)

        steps_per_episode.append(steps)
        rewards_per_episode.append(total_reward)

        # Update plot
        line.set_xdata(range(1, len(steps_per_episode) + 1))
        line.set_ydata(steps_per_episode)
        line2.set_xdata(range(1, len(rewards_per_episode) + 1))
        line2.set_ydata(rewards_per_episode)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show(block=False)

    # Adjust plot layout
    plt.tight_layout()

    # Save the plot
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    filename = f"screenshots/blocksworld_qlearning_for_{run_name}.png"
    fig.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close('all')

    print("Training complete ✅. Exiting now.")

    # Finally close the environment
    env.close()

# Constant Environments
ENV_WITH_3_DIGIT_STATE = gymnasium.make("blocksworld_env/BlocksWorld-v0", render_mode="human")
ENV_WITH_6_DIGIT_STATE = gymnasium.make("blocksworld_env/BlocksWorldEnvTarget-v0", render_mode="human")

ENV = ENV_WITH_3_DIGIT_STATE # To Switch between Environments
# ENV = ENV_WITH_6_DIGIT_STATE # to use the 6-digit state environment

# Hyperparameter sets
SET1 = {
    "episodes": 30,  # Number of episodes the agent will train for
    "gamma": 0.9, # Discount factor for balance between short-term and long-term rewards
    "epsilon": 0.2, # Epsilon for exploration-exploitation trade-off
    "epsilon_min": 0.01, # Minimum epsilon value to ensure some exploration
    "decay": 0.01, # Decay rate for epsilon to reduce exploration over time
    "alpha": 0.5 # Learning rate for updating Q-values
}
SET2 = {
    "episodes": 30,
    "gamma": 0.85,         # Lower discounting — more short-term focused
    "epsilon": 0.3,        # Higher exploration initially
    "epsilon_min": 0.05,   # Allow more exploration even at the end
    "decay": 0.02,         # Faster epsilon decay
    "alpha": 0.6           # More aggressive learning
}

SET3 = {
    "episodes": 30,
    "gamma": 0.99,         # Very long-term focused
    "epsilon": 0.1,        # More exploitation from the start
    "epsilon_min": 0.01,   # Still allow some exploration
    "decay": 0.005,        # Very slow decay
    "alpha": 0.3           # Conservative learning rate
}

# Main function to run the training
def main():
    try:
        train_qlearning(ENV, **SET1, run_name="DEMO RUN")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        os._exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected exception: {e}")
    finally:
        print("[INFO] Cleaning up...")
        try:
            ENV.close()
            print("[INFO] Environment closed successfully.")
        except Exception as e:
            print(f"[WARN] Failed to close environment: {e}")
        os._exit(0)

if __name__ == "__main__":
    main()