from gym import envs

envs_names = [ env.id for env in envs.registry.all() ]

for name in sorted(envs_names):
  print(name)