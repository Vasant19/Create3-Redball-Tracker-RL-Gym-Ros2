import gymnasium
import gymnasium_environments
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

def train_qlearning(env, episodes, gamma, epsilon, epsilon_min, decay, alpha, run_name="default"):
    # Validate discrete observation and action space
    try:
        numstates = env.observation_space.n
        numactions = env.action_space.n
    except AttributeError:
        raise ValueError("Environment must use Discrete observation and action spaces.")

    # Initialize Q-table
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
    ax.set_title(f'Q-Learning on RedBall for {run_name}', fontsize=12)
    ax.grid(True)
    plt.legend()

    # Prepare log file
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
        last_valid_state = 320 if state is None else state

        while not done:
            os.system('clear')
            print(f"Episode {i+1} / {episodes}")

            steps += 1

            # Replace None state if needed
            if state is None:
                state = last_valid_state
            else:
                last_valid_state = state

            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = qtable[state].index(max(qtable[state]))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Handle case where red ball is lost
            if next_state is None:
                reward = -1.0
                next_state = state
            else:
                last_valid_state = next_state

            # Q-learning update
            qtable[state][action] = (1 - alpha) * qtable[state][action] + alpha * (
                reward + gamma * max(qtable[next_state])
            )

            total_reward += reward
            state = next_state

        # Log episode result
        with open(log_filename, "a") as f:
            f.write(f"Episode {i+1}: Steps {steps}, Total Reward {total_reward:.2f}\n")

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
    plt.tight_layout()

    # Save plot
    os.makedirs("screenshots", exist_ok=True)
    plot_file = f"screenshots/qlearning_redball_{run_name}.png"
    fig.savefig(plot_file)
    print(f"Saved plot to {plot_file}")
    plt.close('all')

    print("Training complete ✅. Exiting now.")
    env.close()

# -- Environment and Hyperparameters --

ENV = gymnasium.make("gymnasium_environments/CreateRedBall-v0", render_mode="human")

SET1 = {
    "episodes": 200,
    "gamma": 0.99,
    "epsilon": 0.5,
    "epsilon_min": 0.01,
    "decay": 0.5,
    "alpha": 0.1
}

def main():
    try:
        train_qlearning(ENV, **SET1, run_name="RedBall-Baseline")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        os._exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected exception: {e}")
    finally:
        print("[INFO] Cleaning up...")
        try:
            ENV.close()
        except Exception as e:
            print(f"[WARN] Failed to close environment: {e}")
        os._exit(0)

if __name__ == "__main__":
    main()
