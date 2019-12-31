import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
import logging
import sys


def RL_test():
    env = gym.make('CartPole-v0')
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    obs = env.reset()
    # env.render()
    print('initial observation:', obs)

    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    print('action:', action)
    print('next observation:', obs)
    print('reward:', r)
    print('done:', done)
    print('info:', info)

    obs_size = env.observation_space.shape[0]
    print('obs_size:', obs_size)
    n_actions = env.action_space.n
    print('action_size:', n_actions)

    _q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions,
        n_hidden_layers=2, n_hidden_channels=50)

    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(_q_func)

    gamma = 0.95

    explore = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.action_space.sample)

    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    phi = lambda x: x.astype(np.float32, copy=False)

    agent = chainerrl.agents.DoubleDQN(q_function=_q_func,
                                       replay_buffer=replay_buffer,
                                       optimizer=optimizer,
                                       gamma=gamma,
                                       explorer=explore,
                                       replay_start_size=500,
                                       update_interval=1,
                                       target_update_interval=100,
                                       phi=phi)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    chainerrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=2000,
        eval_n_steps=None,
        eval_n_episodes=10,
        train_max_episode_len=200,
        eval_interval=1000,
        outdir='result'
    )

    # n_episodes = 280
    # max_episode_len = 200
    # for i in range(1, n_episodes + 1):
    #     obs = env.reset()
    #     reward = 0
    #     done = False
    #     R = 0
    #     t = 0
    #     while not done and t < max_episode_len:
    #         action = agent.act_and_train(obs, reward)
    #         obs, reward, done, _ = env.step(action)
    #         R += reward
    #         t += 1
    #     if i % 10 == 0:
    #         print('episode:', i,
    #               'R:', R,
    #               'statistics:', agent.get_statistics())
    #     agent.stop_episode_and_train(obs, reward, done)
    # print('Finished.')
    #
    # print('++++++++++++++++++++++++++++++++++')
    # for i in range(10):
    #     obs = env.reset()
    #     done = False
    #     R = 0
    #     t = 0
    #     while not done and t < 200:
    #         action = agent.act(obs)
    #         obs, reward, done, _ = env.step(action)
    #         R += r
    #         t += 1
    #     print('test episode:', i, 'R:', R)
    #     agent.stop_episode()


if __name__ == "__main__":
    RL_test()
