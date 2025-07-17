import os
import gymnasium
import blocksworld_env
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordEpisodeStatistics
from helper_callback import EpisodeLoggerCallback 

# Prepare environment and wrap with episode statistics wrapper
env = gymnasium.make("blocksworld_env/BlocksWorld-v0", render_mode=None)
env = RecordEpisodeStatistics(env) 

# Instantiate the model
model = DQN(
    "MlpPolicy",
    env,
    verbose=2,
    learning_rate=5e-4,            # Slightly lower LR for more stable updates
    buffer_size=20000,             # Moderate replay buffer
    learning_starts=500,           # Start training after 500 steps
    batch_size=64,                 # Larger batch size for stability
    gamma=0.98,                    # High discount factor (long-term rewards)
    train_freq=1,                  # Train every step for faster learning feedback
    target_update_interval=500,    # Update target network more frequently
    exploration_fraction=0.2,      # Explore for first 20% of training
    exploration_final_eps=0.02,    # Final low epsilon for mostly greedy actions
    max_grad_norm=10,              # Gradient clipping to stabilize training
)

# Create callback instance
log_path = "./logs/training_log_DQN.txt"
callback = EpisodeLoggerCallback(log_path, verbose=1)

# Train the model 
try:
    model.learn(total_timesteps=30000, callback=callback, log_interval=4)
    print("✅ DQN Model trained successfully!")

# Handle exceptions
except KeyboardInterrupt:
    print("⛔ Training interrupted by user!")
except Exception as e:
    print(f"❌ An error occurred during training: {e}")

# Save the model
finally:
    model.save("./models/dqn_blocksworld")
    print("✅ DQN Model saved successfully!")
    env.close()
    os._exit(0)