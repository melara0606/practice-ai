import gym
from gym.spaces import *
import sys

def print_spaces(space):
  print(space)
  if isinstance(space, Box):
    print("\n Cota inferior: ", space.low)
    print("\n Cota superior: ", space.high)
  

if __name__ == '__main__':
  enviroment = gym.make(sys.argv[1])
  print("Espacio de observaciones: ")
  print_spaces(enviroment.observation_space)
  print("Espacio de acciones: ")
  print_spaces(enviroment.action_space)

  try:
    print("Descrpcion de las acciones: ", enviroment.unwrapped.get_action_meanings())
  except AttributeError:
    pass