from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library

standard_library.install_aliases()  # NOQA

from args import args
import os
import sys

from chainer import optimizers
import gym
from gym import spaces
import numpy as np
import chainer

import chainerrl
from chainerrl.agents.dqn import DQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer


class DQN2NSM(object):
    def __init__(self):
        print()

    def env_make(self, test=True):
        """
        environment
        :param test:
        :return:
        """
        env = gym.make(args.env)
        print(args.env)
        print(env)
        print(env.action_space.sample())
        print('action-space.low-high', env.action_space.low, env.action_space.high)
        print(spaces.box)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        print(env)
        return env
        pass

    def net_def(self):
        env = self.env_make(test=False)
        timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        obs_space = env.observation_space
        obs_size = obs_space.low.size
        action_space = env.action_space

        n_actions = action_space.n
        q_func = q_functions.FCStateQFunctionWithDiscreteAction(obs_size, n_actions,
                                                                n_hidden_channels=args.n_hdden_channels,
                                                                n_hidden_layers=args.n_hidden_layers)
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(args.start_epsilon,
                                                      args.end_epsilon,
                                                      args.final_exploration_steps,
                                                      action_space.sample)
        pass


if __name__ == "__main__":
    dqn = DQN2NSM()
    dqn.env_make()
