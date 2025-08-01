import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.verbose = verbose
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("Episode\tSteps\tReward\n")

        # Initialize lists to store steps and rewards
        self.episode_rewards = []
        self.episode_lengths = []

        # Setup live plot
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.reward_line, = self.ax.plot([], [], label='Episode Reward', color='green')
        self.step_line, = self.ax.plot([], [], label='Episode Steps', color='blue')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Dqn Training Progress')
        self.ax.grid(True)
        self.ax.legend()
        plt.ion()
        plt.show()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                steps = ep_info["l"]
                reward = ep_info["r"]
                
                self.episode_lengths.append(steps)
                self.episode_rewards.append(reward)

                # Log to file
                with open(self.log_path, "a") as f:
                    f.write(f"{len(self.episode_rewards)}\t{steps}\t{reward:.2f}\n")

                # Console log
                if self.verbose > 0:
                    print(f"Episode {len(self.episode_rewards)} ended: Steps={steps}, Reward={reward:.2f}")

                # Update plot
                x = list(range(1, len(self.episode_rewards) + 1))
                self.reward_line.set_data(x, self.episode_rewards)
                self.step_line.set_data(x, self.episode_lengths)
                self.ax.relim()
                self.ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)

        return True

    def _on_training_end(self):
        plt.ioff()
        plt.tight_layout()
        os.makedirs("screenshots", exist_ok=True)
        save_path = os.path.join("screenshots", "dqn_training_plot.png")
        self.fig.savefig(save_path)
        print(f"📊 Final training plot saved to {save_path}")
        plt.close()
