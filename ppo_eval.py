import gymnasium
import gymnasium_environments
from stable_baselines3 import PPO
env = gymnasium.make("gymnasium_environments/CreateRedBall-v0", render_mode="human")

model = PPO.load("./models/ppo_redball", env=env)  # load saved model with environment
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