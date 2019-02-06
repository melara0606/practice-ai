import gym

environment = gym.make('Acrobot-v1')
environment.reset()

for _ in range(2000):
  environment.render()
  environment.step( environment.action_space.sample() )

environment.close()