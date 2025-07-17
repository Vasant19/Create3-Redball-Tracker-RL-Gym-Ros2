import gymnasium
import blocksworld_env
from stable_baselines3 import PPO
env = gymnasium.make("blocksworld_env/BlocksWorld-v0", render_mode="human")

model = PPO.load("./models/ppo_blocksworld", env=env)  # load saved model with environment
obs, info = env.reset()
try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()
        print(f"Action: {action}, Reward: {reward}")
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    print("Evaluation interrupted by user.")
except Exception as e:
    print(f"An error occurred during evaluation: {e}")
finally:
    env.close()
    print("Evaluation completed successfully! ✅")