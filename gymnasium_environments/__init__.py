from gymnasium.envs.registration import register

register(
    id="gymnasium_environments/GridWorld-v0",
    entry_point="gymnasium_environments.envs:GridWorldEnv",
)

register(
    id="gymnasium_environments/CreateRedBall-v0",
    entry_point="gymnasium_environments.envs:CreateRedBallEnv"
)