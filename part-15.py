import numpy as np
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)


class Bandit:
    def __init__(self, number, arms_probability):
        self.number = number
        self.arms_probability = arms_probability
        self.pull_stats = [0] * len(arms_probability)

    @property
    def number_of_arms(self):
        return len(self.arms_probability)

    def pull_arm(self, number):
        self.pull_stats[number] += 1

        result = np.random.randn(1)

        if result > self.arms_probability[number]:
            return 1

        return -1


class Agent:
    def __init__(self, learning_rate, e, number_of_states, number_of_actions):
        self.learning_rate = learning_rate
        self.e = e

        self.state_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        state_one_hot = tf.one_hot(indices=self.state_holder,
                                   depth=number_of_states)

        self.model = self._build_model(state_one_hot, number_of_actions)
        self.output = tf.squeeze(self.model, axis=0)
        self.chosen_action = tf.argmax(self.output)

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

        responsible_weight = self.output[tf.squeeze(self.action_holder)]
        loss = -(tf.log(responsible_weight) * self.reward_holder)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.update = optimizer.minimize(loss)

    def _build_model(self, x, number_of_outputs):
        model = tf.layers.Dense(
            number_of_outputs,
            use_bias=False,
            kernel_initializer=tf.initializers.ones(),
            activation=tf.nn.relu)(x)

        # model = tf.layers.Dense(len(bandits), use_bias=False,
        #     kernel_initializer=
        #         tf.initializers.ones())(x)
        # model = tf.Variable(tf.random_uniform([num_bandits], 0, 0.01))
        # model = tf.matmul(x, model)

        return model

    def play(self, bandit, tf_session):
        if np.random.rand(1) < self.e:
            action = np.random.randint(bandit.number_of_arms)
        else:
            action = sess.run(self.chosen_action,
                    feed_dict={self.state_holder: [bandit.number]})

        # print(output)
        # print("state: {}, action: {}".format(bandit.number, action))
        reward = bandit.pull_arm(action)
        # print("reward: {}".format(reward))


        _ = sess.run([self.update],
                feed_dict={
                    self.state_holder: [bandit.number],
                    self.action_holder: [action],
                    self.reward_holder: [reward]
                })

    def get_weights(self, state, tf_session):
        output = sess.run(self.output,
                    feed_dict={self.state_holder: [state]})
        return output


    def adapt(self, state, reward, tf_session):
        pass





bandits_probs = [[0.2,0,-0.0,-5], [0.1,-5,1,0.25], [-5,5,5,5]]
bandits = []
agent = Agent(learning_rate=0.001,
              e=0.1,
              number_of_states=len(bandits_probs),
              number_of_actions=len(bandits_probs[0]))

for i, probabilities in enumerate(bandits_probs):
    bandits.append(Bandit(i, probabilities))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(1000):
        chosen_bandit_number = np.random.randint(0, len(bandits))
        agent.play(bandits[chosen_bandit_number], sess)


    print("------------------")
    for i, bandit in enumerate(bandits):
        print("bandit {}: {}".format(i, agent.get_weights(i, sess)))
        print("  The best arm is: {}".format(
            np.argmax(agent.get_weights(i, sess))))
