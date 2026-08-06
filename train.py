from envs.exploration_env import ExplorationEnv

if __name__ == "__main__":
    env = ExplorationEnv()
    obs, shared_obs, info = env.reset()
    print(obs.shape) # reset完全成功