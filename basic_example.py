# Example from http://docs.gym.derkgame.com/#installing

from gym_derk.envs import DerkEnv

env = DerkEnv(turbo_mode=False, app_args={"window_size": (1000, 1000), "browser_logs": True, "chrome_devtools": True}, agent_server_args={"port": 8888})

for t in range(3):
  print("Running episode")
  observation_n = env.reset()
  
  while True:
    action_n = [env.action_space.sample() for i in range(env.n_agents)]
    print(action_n)
    observation_n, reward_n, done_n, info = env.step(action_n)
    print(observation_n, reward_n, done_n, info)
    if all(done_n):
      print("Episode finished")
      break
  print(env.episode_stats)      
env.close()

