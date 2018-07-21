import numpy as np
import tensorflow as tf
import gym
import pprint
import argparse
import signal
import io
import imageio

import matplotlib.pyplot as plt


tf.set_random_seed(0)
np.random.seed(0)

RENDER_EVERY_EPISODES = 200


class Supervisor:
    def __init__(self,
                 average_windows_size):
        self.average_windows_size = average_windows_size

        self.episodes_reward = []
        self.average_reward = 0

        self.episode_count = -1

        self.last_episode_render = None

    def _update_average_reward(self):
        last_reward_samples = self.episodes_reward[
                                    -self.average_windows_size:]
        self.average_reward = (np.sum(last_reward_samples)
                                    / len(last_reward_samples))

    def episode_reward(self, reward):
        self.episodes_reward.append(reward)
        self._update_average_reward()

    @property
    def last_episode_reward(self):
        return self.episodes_reward[-1]

    def episode_start(self):
        self.last_episode_render = None
        self.episode_count += 1

    def episode_done(self):
        pass

    def summary(self):
        print(("\rEpisode {:>6}"
                + " | Average reward: {:>6.0f}").format(
                    self.episode_count,
                    self.average_reward), end="")


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
    def __init__(self, learning_rate, number_of_states, number_of_actions):
        self.learning_rate = learning_rate

        self.experience = ExperienceTrace(0.99)
        self.state = None

        self.supervisor = Supervisor(average_windows_size=1000)

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
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) 
                                        * self.reward_holder)

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

    def play(self, env, tf_session):
        if self.state is None:
            self.state = env.reset()

            self.render_frames = []

            self.supervisor.episode_start()


        output = tf_session.run(self.output,
                    feed_dict={self.state_holder: [self.state]})
        squeezed_output = np.squeeze(output)
        chosen_output_value = np.random.choice(squeezed_output,
                                               p=squeezed_output)
        action = np.argmax(output == chosen_output_value)

        # print("Action: {}".format(action))

        if (self.supervisor.episode_count % RENDER_EVERY_EPISODES) == 0:
            self.render_frames.append(env.render(mode='rgb_array'))

        state, reward, done, info = env.step(action)

        self.experience.add(self.state, action, reward)

        if done:
            if (self.supervisor.episode_count % 1) == 0:
                states = self.experience.states
                actions = self.experience.actions
                rewards = self.experience.discounted_sum_rewards

                # pprint.pprint(states)
                # pprint.pprint(actions)
                # pprint.pprint(rewards)

                self.supervisor.episode_reward(np.sum(rewards))

                self.supervisor.summary()

                _, loss = tf_session.run(
                        [self.update, self.loss],
                        feed_dict={
                            self.state_holder: states,
                            self.action_holder: actions,
                            self.reward_holder: rewards
                        })

                # pprint.pprint(loss)

                self.experience.flush()

            if self.render_frames:
                self.supervisor.last_episode_render = io.BytesIO()
                imageio.mimsave(self.supervisor.last_episode_render,
                                self.render_frames,
                                format='GIF',
                                fps=60)

            self.state = None
            self.supervisor.episode_done()

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

interactive_mode = False

def receive_signal(signum, stack):
    global interactive_mode

    interactive_mode = not interactive_mode

signal.signal(signal.SIGUSR1, receive_signal)

agent = Agent(learning_rate=0.001,
              number_of_states=4,
              number_of_actions=2)

gym.envs.register(
    id='CartPoleMoreSteps-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=None,
    reward_threshold=None
)
env = gym.make('CartPoleMoreSteps-v0')
# env = gym.make('CartPole-v0')
env.seed(0)

init = tf.global_variables_initializer()

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
# config = None
with tf.Session(config=config) as sess:
    sess.run(init)

    writer = tf.summary.FileWriter("logs", sess.graph)

    while True:
        while agent.play(env, sess):
            pass

        summary = tf.Summary()

        summary.value.add(tag="reward",
            simple_value=agent.supervisor.last_episode_reward)
        summary.value.add(tag="reward (average)",
            simple_value=agent.supervisor.average_reward)

        if agent.supervisor.last_episode_render:
            summary.value.add(tag="reward (image)",
                simple_value=agent.supervisor.last_episode_reward)

            image = tf.Summary.Image()
            image.encoded_image_string = (
                agent.supervisor.last_episode_render.getvalue()
            )
            summary.value.add(tag="render", image=image)

        writer.add_summary(summary, agent.supervisor.episode_count)

        if args.episodes < 0:
            continue
        elif agent.supervisor.episode_count >= args.episodes:
            writer.close()
            break

    print()
