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
from NSMGame import Game
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

    def clip_action_filter(self, a, env):
        return np.clip(a, env.action_space.low, env.action_space.high)

    def env_make(self, test=True):
        """
        environment
        :param test:
        :return:
        """
        env = Game()
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, self.clip_action_filter)
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

    def main(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)

        # Set a random seed used in ChainerRL
        misc.set_random_seed(args.seed, gpus=(args.gpu,))

        args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
        print('Output files are saved in {}'.format(args.outdir))

        env = self.env_make(test=False)
        timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        obs_space = env.observation_space
        obs_size = obs_space.low.size
        action_space = env.action_space

        if isinstance(action_space, spaces.Box):
            action_size = action_space.low.size
            # Use NAF to apply DQN to continuous action spaces
            q_func = q_functions.FCQuadraticStateQFunction(
                obs_size, action_size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
                action_space=action_space)
            # Use the Ornstein-Uhlenbeck process for exploration
            ou_sigma = (action_space.high - action_space.low) * 0.2
            explorer = explorers.AdditiveOU(sigma=ou_sigma)
        else:
            n_actions = action_space.n
            q_func = q_functions.FCStateQFunctionWithDiscreteAction(
                obs_size, n_actions,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers)
            # Use epsilon-greedy for exploration
            explorer = explorers.LinearDecayEpsilonGreedy(
                args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
                action_space.sample)

        if args.noisy_net_sigma is not None:
            links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
            # Turn off explorer
            explorer = explorers.Greedy()

        # Draw the computational graph and save it in the output directory.
        chainerrl.misc.draw_computational_graph([q_func(np.zeros_like(obs_space.low, dtype=np.float32)[None])],
                                                os.path.join(args.outdir, 'model'))

        opt = optimizers.Adam()
        opt.setup(q_func)

        rbuf = self.buffer()

        agent = DQN(q_func, opt, rbuf, gamma=args.gamma, explorer=explorer, replay_start_size=args.replay_start_size,
                    target_update_interval=args.target_update_interval,
                    update_interval=args.update_interval,
                    minibatch_size=args.minibatch_size,
                    target_update_method=args.target_update_method,
                    soft_update_tau=args.soft_update_tau)
        if args.load:
            agent.load(args.load)

        eval_env = self.env_make(test=True)

        if args.demo:
            eval_stats = experiments.eval_performance(
                env=eval_env,
                agent=agent,
                n_steps=None,
                n_episodes=args.eval_n_runs,
                max_episode_len=timestep_limit
            )
            print('n_runs: {} mean: {} median: {} stdev: {}'.format(
                args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']
            ))
        else:
            experiments.train_agent_with_evaluation(
                agent=agent, env=env, steps=args.steps,
                eval_n_steps=None,
                eval_n_episodes=args.eval_n_runs, eval_interval=args.eval_interval,
                outdir=args.outdir, eval_env=eval_env,
                train_max_episode_len=timestep_limit
            )
        pass

    def buffer(self):
        rbuf_capacity = 5 * 10 * 5
        if args.minibatch_size is None:
            args.minibatch_size = 32
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                        // args.update_interval
            return replay_buffer.PrioritizedReplayBuffer(rbuf_capacity, betasteps=betasteps)
        else:
            return replay_buffer.ReplayBuffer(rbuf_capacity)


if __name__ == "__main__":
    dqn = DQN2NSM()
    dqn.env_make()
