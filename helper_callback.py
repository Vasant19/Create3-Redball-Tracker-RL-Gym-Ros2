import os
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback to log episode statistics
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("Episode\tSteps\tReward\n")  

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                with open(self.log_path, "a") as f:
                    f.write(f"{self.num_timesteps}\t{ep_info['l']}\t{ep_info['r']}\n")
                if self.verbose > 0:
                    print(f"Episode ended: TimeSteps={ep_info['l']} reward={ep_info['r']}")
        return True