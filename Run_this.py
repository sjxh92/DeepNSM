from DQN import DQNPrioiritizedReplay
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

MEMORY_SIZE = 10000


def run_deepnsm(RL):
    total_step = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        # initial observation
        observation = env.init_state()
        print('--------------------step------------------------')
        print(observation)
        while True:
            # the exploration step of a episode equals the number of requests
            # request = 0

            # RL choose the action based on the observation
            action = RL.choose_action(observation)

            # RL execute action
            observation_, reward = env.next(action)

            print('---------------to the store transition--------------')
            print(observation)
            print(action)
            print(reward)
            print(observation_)
            # print(action.shape)

            # store the transition(s, a, r, s_)
            RL.store_transition(observation, action, reward, observation_)

            if total_step > MEMORY_SIZE:
                RL.learn()

            if total_step % (2 * MEMORY_SIZE) == 0:
                break

            observation = observation_
            total_step += 1
        return RL.q


if __name__ == "__main__":
    env = NetworkEnvironment()
    # print('++++++++++++++++++++++++++++')
    # print(env.topology.nodes.data())
    sess = tf.Session()
    with tf.variable_scope('Natural_DQN'):
        natural_DQN = DQNPrioiritizedReplay(n_actions=env.n_action_mapping, n_features=env.n_feature,
                                            memory_size=MEMORY_SIZE,
                                            e_greedy_increment=0.00005, prioritized=False, sess=sess)
    with tf.variable_scope('DQN_'):
        double_DQN = DQNPrioiritizedReplay(n_actions=env.n_action_mapping, n_features=env.n_feature,
                                           memory_size=MEMORY_SIZE,
                                           e_greedy_increment=0.00005, prioritized=True, sess=sess)

    sess.run(tf.global_variables_initializer())

    q_natural = run_deepnsm(natural_DQN)
    q_double = run_deepnsm(double_DQN)

    print('------------------------------')
    print(len(q_double))
    print(q_natural)

    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
