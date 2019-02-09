import gym

MAX_NUM_EPISODES = 1000

enviroment = gym.make("MountainCar-v0")
for episode in range(MAX_NUM_EPISODES):
  done = False
  obs = enviroment.reset()
  total_reward = 0.0
  step = 0
  
  while not done:
    enviroment.render()
    action = enviroment.action_space.sample()
    next_state, reward, done, info = enviroment.step(action)
    step += 1
    total_reward += reward
    obs = next_state    
  print("\n Episodio numero {} finalizado con {} iteraciones. Recompensa final = {}.".format(episode, step + 1, total_reward))

enviroment.close()
