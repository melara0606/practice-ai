import gym
import numpy as np

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

class QLearner(object):
  def __init__(self, enviroment):
    self.obs_shape = enviroment.observation_space.shape
    self.obs_high = enviroment.observation_space.high
    self.obs_low = enviroment.observation_space.low
    self.obs_bins = NUM_DISCRETE_BINS
    self.obs_width = ( self.obs_high-self.obs_low ) / self.obs_bins

    self.action_shape = enviroment.action_space.n
    self.Q = np.zeros( (self.obs_bins + 1, self.obs_bins + 1 , self.action_shape) )

    self.alpha = ALPHA
    self.gamma = GAMMA
    self.epsilon = 1.0
     
  def discretize(self, obs):
    return tuple(((obs - self.obs_low) / self.obs_width).astype(int))
  def get_action(self, obs):
    discrete_obs = self.discretize(obs)
    if self.epsilon > EPSILON_MIN:
      self.epsilon -= EPSILON_DECAY

    if np.random.random() > self.epsilon:
      return np.argmax(self.Q[discrete_obs]) 
    else:
      return np.random.choice([ a for a in range(self.action_shape) ])
  def learn(self, obs, action, reward, next_obs):
    discrete_obs = self.discretize(obs)
    discrete_next_obs = self.discretize(next_obs)
    
    td_target = reward + self.gamma * np.max( self.Q[discrete_next_obs] )
    td_error = td_target  - self.Q[discrete_obs][action]
    self.Q[ discrete_obs ][ action ] += self.alpha * td_error



def train(agent, enviroment):
  best_reward = -float('inf')
  for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = enviroment.reset()
    total_reward = 0.0
    while not done:
      action = agent.get_action(obs)
      next_obs, reward, done, info = enviroment.step(action)
      agent.learn(obs, action, reward, next_obs)
      obs = next_obs
      total_reward += reward
    
    if total_reward > best_reward:
      best_reward = total_reward
    
    print("Episodio numero {} con recompesa {} mejor recompesa {}, epsilon {}".format(episode, total_reward, best_reward, agent.epsilon))
  
  return np.argmax(agent.Q, axis= 2)


def test(agent, enviroment, policy):
  done = False
  obs = enviroment.reset()
  total_reward = 0.0

  while not done:
    action = policy[agent.discretize(obs)]
    next_obs, reward, done, info = enviroment.step(action)
    obs = next_obs
    total_reward += reward
  return total_reward


if __name__ == "__main__":
  enviroment = gym.make("MountainCar-v0")
  agent = QLearner(enviroment)
  learned_policy = train(agent, enviroment)
  monitor_path = "./monitor_output"
  enviroment = gym.wrappers.Monitor(enviroment, monitor_path, force= True)

  for _ in range(1000):
    test(agent, enviroment, learned_policy)  
  enviroment.close() 