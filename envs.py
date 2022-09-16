import gym
import gym_minigrid
import numpy as np

class SingleEnvWrapper():
    def __init__(self, env, seed=False):
        self._env = env
        self.seed = seed
        if isinstance(env.observation_space, gym.spaces.Dict) == False or "image" not in env.observation_space.spaces.keys():
            print('not minigrid obs')
            obs_dim = env.observation_space.shape[0]
            obs_dim += 2
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        else:
            self.observation_space = self._env.observation_space.spaces["image"]

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # torso_height, torso_ang = self._env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        # obs = np.append(obs, [torso_height, torso_ang])

        return obs['image'], reward, done, info
        # obs = np.concatenate((obs['image'].reshape((-1)), np.array([True], dtype=np.int)), -1)
        # return obs, reward, done, info


    def reset(self):
        if self.seed:
            obs = self._env.reset(self.seed)
        else:
            obs = self._env.reset()
        return obs['image']

class OneHotAction:
    
  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    # space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference