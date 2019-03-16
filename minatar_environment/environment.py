################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
from importlib import import_module
import numpy as np
import sys
import gym
from gym import spaces


#####################################################################################################################
# Environment
#
# Wrapper for all the specific game environments. Imports the environment specified by the user and then acts as a
# minimal interface. Also defines code for displaying the environment for a human user. 
#
#####################################################################################################################

class GymEnvironment(gym.Env):
    def __init__(self, minatar_env):
        self._env = minatar_env
        self.action_space = spaces.Discrete(len(minatar_env.env.action_map))
        channels = minatar_env.env.channels

        high = sys.maxsize
        low = -sys.maxsize - 1
        nchannels = 0
        for c, v in channels.items():
            if v > high:
                high = v
            if v < low:
                low = v
            nchannels += 1

        self.nchannels = nchannels
        self.observation_space = spaces.Box(low=low, high=high, shape=(10, 10, 1))

    def collapse(self, state):
        ns = np.zeros((10, 10), dtype=np.int)

        for i in range(self.nchannels):
            c = (state[:,:,i] == 1)
            #print(c)
            #print(ns)
            ns[c] = i + 1
        return np.expand_dims(ns, axis=2)

    def scale(self, state):
        return (state - self.nchannels / 2.0) / self.nchannels

    def step(self, action):
        r, terminal = self._env.act(action)
        state = self._env.state()
        state = self.collapse(state)
        state = self.scale(state)
        return state, r, terminal, {}

    def reset(self):
        self._env.reset()
        state = self._env.state()
        state = self.collapse(state)
        state = self.scale(state)
        return state

    def close(self):
        pass

    def get_action_meanings(self):
        return self._env.env.action_map



def Make(name, seed):
    env = Environment(name, sticky_action_prob=0, random_seed=seed)
    return GymEnvironment(env)



class Environment:
    def __init__(self, env_name, sticky_action_prob = 0.1, difficulty_ramping = True, random_seed = None):
        env_module = import_module('minatar.environments.'+env_name)
        self.env_name = env_name
        self.env = env_module.Env(ramping = difficulty_ramping, seed = random_seed)
        self.n_channels = self.env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0

    def step(self, action):
        self.act(action)

    # Wrapper for env.act
    def act(self, a):
        if(np.random.rand()<self.sticky_action_prob):
            a = self.last_action
        self.last_action = a
        return self.env.act(a)

    # Wrapper for env.state
    def state(self):
        return self.env.state()

    # Wrapper for env.reset
    def reset(self):
        return self.env.reset()

    # Wrapper for env.state_shape
    def state_shape(self):
        return self.env.state_shape()

    # All MinAtar environments have 6 actions
    def num_actions(self):
        return 6

    def game_name(self):
        return self.env_name
