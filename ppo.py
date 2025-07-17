import os
import gymnasium
import blocksworld_env
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordEpisodeStatistics
from helper_callback import EpisodeLoggerCallback 

# Prepare environment and wrap with episode statistics wrapper
env = gymnasium.make("blocksworld_env/BlocksWorld-v0", render_mode=None)
env = RecordEpisodeStatistics(env) 

# Instantiate the model
model = PPO(
    "MlpPolicy",
    env,
    verbose=2,
    learning_rate=5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.98,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
)

# Create callback instance
log_path = "./logs/training_log_PPO.txt"
callback = EpisodeLoggerCallback(log_path, verbose=1)

# Train the model 
try:
    model.learn(total_timesteps=30000, callback=callback, log_interval=4)
    print("✅ PPO Model trained successfully!")

# Handle exceptions
except KeyboardInterrupt:
    print("⛔ Training interrupted by user!")
except Exception as e:
    print(f"❌ An error occurred during training: {e}")

# Save the model
finally:
    model.save("./models/ppo_blocksworld")
    print("✅ PPO Model saved successfully!")
    env.close()
    os._exit(0)