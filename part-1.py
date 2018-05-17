import numpy as np
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)


# List out our bandits.
# Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)

def pull_bandit(bandit):
    # Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return [1]
    else:
        #return a negative reward.
        return [-1]


def build_model(x):
    # model = tf.layers.Dense(10, activation=tf.nn.relu)(x)
    model = tf.layers.Dense(len(bandits), use_bias=False,
        kernel_initializer=
            tf.initializers.ones())(x)
    # model = tf.Variable(tf.random_uniform([16,4],0,0.01))
    # model = tf.matmul(x, model)
    return model


reward_holder = tf.placeholder(shape=[1, 1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1, 1], dtype=tf.int32)

# with tf.Session() as sess:
#     print(sess.run(tf.one_hot(indices=[0], depth=16)))
#     print(sess.run(tf.zeros([1,1])))
# exit(0)

model = build_model(tf.ones([1,1]))
chosen_action = tf.argmax(input=model, axis=1)

# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(tf.squeeze(model))
#     print(action_holder)
# exit(0)

responsible_weight = model.slice(action_holder FIXME
loss = -(tf.log(responsible_weight) * reward_holder)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update_model = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()



total_episodes = 10000 # Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits) # Set scoreboard for bandits to 0.
e = 0.1 # Set the chance of taking a random action.

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs', sess.graph)

    i = 0
    while i < total_episodes:
        # Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            action = [np.random.randint(num_bandits)]
        else:
            action = sess.run(chosen_action)

        reward = pull_bandit(bandits[action])

        # Update the network.
        _, resp, ww = sess.run([update_model, responsible_weight, model],
                feed_dict={reward_holder: [reward], action_holder: [action]})

        # Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the {} bandits: {}".format(
                                        num_bandits, total_reward))
        i+=1

    writer.close()

print(ww)
print("The agent thinks bandit {} is the most promising".format(
                                                    np.argmax(ww, 1)+1))
if np.argmax(ww, 1) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
