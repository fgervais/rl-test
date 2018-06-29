import numpy as np
import tensorflow as tf
import gym
import pprint
import argparse

tf.set_random_seed(0)
np.random.seed(0)


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

        self.reward_per_episode = []
        self.episode_number = 1

        self.watch_episodes = 10
        self.watch_reward_threshold = 0

        self.state_holder = tf.placeholder(shape=[None, number_of_states],
                                            dtype=tf.float32)

        self.model = self._build_model(self.state_holder, number_of_actions)
        # self.output = tf.squeeze(self.model, axis=0)
        self.output = self.model
        self.chosen_action = tf.argmax(self.output, axis=1)

        self.reward_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None, 1], dtype=tf.int32)

        row_index = tf.range(tf.shape(self.action_holder)[0])[:, tf.newaxis]
        full_indices = tf.stack([row_index, self.action_holder], axis=2)
        self.responsible_outputs = tf.gather_nd(self.output, full_indices)

        # self.loss = -(tf.log(responsible_outputs) * self.reward_holder)
        self.policy = tf.log(self.responsible_outputs) * self.reward_holder
        self.loss = -tf.reduce_mean(self.policy)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.update = optimizer.minimize(self.loss)

    def _build_model(self, x, number_of_outputs):
        model = tf.layers.Dense(
            8,
            use_bias=False,
            # kernel_initializer=tf.initializers.ones(),
            activation=tf.nn.relu)(x)
        model = tf.layers.Dense(
            number_of_outputs,
            use_bias=False,
            # kernel_initializer=tf.initializers.ones(),
            activation=tf.nn.softmax)(model)

        # model = tf.layers.Dense(len(bandits), use_bias=False,
        #     kernel_initializer=
        #         tf.initializers.ones())(x)
        # model = tf.Variable(tf.random_uniform([num_bandits], 0, 0.01))
        # model = tf.matmul(x, model)

        return model

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def play(self, env, tf_session):
        if self.state is None:
            self.state = env.reset()


        # output = tf_session.run(self.output,
        #             feed_dict={self.state_holder: [self.state, self.state]})
        # print(output)
        # exit(0)


        # if np.random.rand(1) < self.e:
        #     action = env.action_space.sample()
        # else:
        #     # print("Using action from model")
        #     action = tf_session.run(self.chosen_action,
        #             feed_dict={self.state_holder: [self.state]})
        #     action = np.squeeze(action)

        # a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
        output = tf_session.run(self.output,
                    feed_dict={self.state_holder: [self.state]})
        squeezed_output = np.squeeze(output)
        chosen_output_value = np.random.choice(squeezed_output,
                                               p=squeezed_output)
        action = np.argmax(output == chosen_output_value)

        # print("Action: {}".format(action))
        if self.watch_episodes:
            env.render()
        state, reward, done, info = env.step(action)

        self.experience.add(self.state, action, reward)

        if done:
            print("done. Did {} steps".format(len(self.experience)))

            if (self.episode_number % 1) == 0:
                states = self.experience.states
                actions = self.experience.actions
                rewards = self.experience.discounted_sum_rewards

                # pprint.pprint(states)
                # pprint.pprint(actions)
                # pprint.pprint(rewards)

                self.reward_per_episode.append(np.sum(rewards))
                running_mean_array = self.running_mean(
                                            self.reward_per_episode, 100)
                if len(running_mean_array) > 0:
                    current_running_mean = running_mean_array[-1]
                else:
                    current_running_mean = 0
                print(current_running_mean)

                if not self.watch_episodes:
                    if current_running_mean > self.watch_reward_threshold:
                        self.watch_episodes = 10
                        self.watch_reward_threshold = current_running_mean * 2


                _, policy, loss = tf_session.run(
                        [self.update, self.policy, self.loss],
                        feed_dict={
                            self.state_holder: states,
                            self.action_holder: actions,
                            self.reward_holder: rewards
                        })

                # pprint.pprint(policy)
                # pprint.pprint(loss)

                self.experience.flush()

                if self.watch_episodes:
                    self.watch_episodes -= 1


            self.state = None
            self.episode_number += 1

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



parser = argparse.ArgumentParser()
parser.add_argument("-e", "--episodes", type=int, required=True)
args = parser.parse_args()

agent = Agent(learning_rate=0.001,
              e=0.3,
              number_of_states=4,
              number_of_actions=2)


gym.envs.register(
    id='CartPoleMoreSteps-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200 * 100,
    reward_threshold=195.0 * 100
)
env = gym.make('CartPoleMoreSteps-v0')
# env = gym.make('CartPole-v0')
env.seed(0)

init = tf.global_variables_initializer()

# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
config = None
with tf.Session(config=config) as sess:
    sess.run(init)

    for _ in range(args.episodes):
        while agent.play(env, sess):
            pass


    # print("------------------")
    # for i, bandit in enumerate(bandits):
    #     print("bandit {}: {}".format(i, agent.get_weights(i, sess)))
    #     print("  The best arm is: {}".format(
    #         np.argmax(agent.get_weights(i, sess))))
