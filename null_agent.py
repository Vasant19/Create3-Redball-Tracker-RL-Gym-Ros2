import gymnasium as gym
import gymnasium_environments
env = gym.make("gymnasium_environments/CreateRedBall-v0", render_mode="human")
observation, info = env.reset()
print(f"Initial Observation: {observation}")

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Observation: {observation}, Reward: {reward}")

    if terminated or truncated:
        observation, info = env.reset()
        print("Environment reset")
        
env.close()