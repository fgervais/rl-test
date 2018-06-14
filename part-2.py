import numpy as np
import tensorflow as tf
import gym
import pprint

tf.set_random_seed(1)
np.random.seed(1)


class ExperienceTrace:
    def __init__(self, discount_factor):
        self.discount_factor = discount_factor

        self.exp = []

    def add(self, state, action, reward):
        self.exp.append([state, action, reward])

    @property
    def states(self):
        # return np.array(self.exp)[:,0]
        return [i[0] for i in self.exp]

    @property
    def actions(self):
        # return np.array(self.exp)[:,1]
        return [[i[1]] for i in self.exp]

    @property
    def discounted_sum_rewards(self):
        discounted_sums = []
        previous_reward_sum = 0
        np_exp = np.array(self.exp)

        for i in reversed(range(len(self.exp))):
            reward_sum = (previous_reward_sum
                            * self.discount_factor
                            + np_exp[i,2])

            discounted_sums.insert(0, [reward_sum])

            previous_reward_sum = reward_sum

        return discounted_sums

    def flush(self):
        self.exp = []

    def __len__(self):
        return len(self.exp)


class Agent:
    def __init__(self, learning_rate, e, number_of_states, number_of_actions):
        self.learning_rate = learning_rate
        self.e = e

        self.experience = ExperienceTrace(0.99)
        self.state = None

        self.state_holder = tf.placeholder(shape=[None, number_of_states],
                                            dtype=tf.float32)

        self.model = self._build_model(self.state_holder, number_of_actions)
        # self.output = tf.squeeze(self.model, axis=0)
        self.output = self.model
        self.chosen_action = tf.argmax(self.output, axis=1)

        self.reward_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None, 1], dtype=tf.int32)

        responsible_weight = self.output[::,self.action_holder]
        loss = -(tf.log(responsible_weight) * self.reward_holder)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.update = optimizer.minimize(loss)

    def _build_model(self, x, number_of_outputs):
        model = tf.layers.Dense(
            8,
            use_bias=False,
            kernel_initializer=tf.initializers.ones(),
            activation=tf.nn.relu)(x)
        model = tf.layers.Dense(
            number_of_outputs,
            use_bias=False,
            kernel_initializer=tf.initializers.ones(),
            activation=tf.nn.softmax)(model)

        # model = tf.layers.Dense(len(bandits), use_bias=False,
        #     kernel_initializer=
        #         tf.initializers.ones())(x)
        # model = tf.Variable(tf.random_uniform([num_bandits], 0, 0.01))
        # model = tf.matmul(x, model)

        return model

    def play(self, env, tf_session):
        if self.state is None:
            self.state = env.reset()


        # output = tf_session.run(self.output,
        #             feed_dict={self.state_holder: [self.state, self.state]})
        # print(output)
        # exit(0)


        if np.random.rand(1) < self.e:
            action = env.action_space.sample()
        else:
            print("Using action from model")
            action = tf_session.run(tf.squeeze(self.chosen_action),
                    feed_dict={self.state_holder: [self.state]})

        #env.render()
        state, reward, done, info = env.step(action)

        self.experience.add(self.state, action, reward)

        if done:
            print("done. Did {} steps".format(len(self.experience)))

            states = self.experience.states
            actions = self.experience.actions
            rewards = self.experience.discounted_sum_rewards

            pprint.pprint(states)
            pprint.pprint(actions)
            pprint.pprint(rewards)

            # for i in range(len(self.experience)):
            _ = tf_session.run(self.update,
                    feed_dict={
                        self.state_holder: states,
                        self.action_holder: actions,
                        self.reward_holder: rewards
                    })

            self.state = None
            self.experience.flush()
            return False
        else:
            self.state = state

        return True

    def get_weights(self, state, tf_session):
        output = tf_session.run(self.output,
                    feed_dict={self.state_holder: [state]})
        return output


    def adapt(self, state, reward, tf_session):
        pass



agent = Agent(learning_rate=1e-2,
              e=0.1,
              number_of_states=4,
              number_of_actions=2)

env = gym.make('CartPole-v0')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(1):
        while agent.play(env, sess):
            pass


    # print("------------------")
    # for i, bandit in enumerate(bandits):
    #     print("bandit {}: {}".format(i, agent.get_weights(i, sess)))
    #     print("  The best arm is: {}".format(
    #         np.argmax(agent.get_weights(i, sess))))
