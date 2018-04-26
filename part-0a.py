import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)


env = gym.make('FrozenLake-v0')
env.seed(1)


def build_model(x):
    # model = tf.layers.Dense(10, activation=tf.nn.relu)(x)
    model = tf.layers.Dense(4, use_bias=False,
        kernel_initializer=
            tf.random_uniform_initializer(minval=0, maxval=0.01))(x)
    # model = tf.Variable(tf.random_uniform([16,4],0,0.01))
    # model = tf.matmul(x, model)
    return model

state = tf.placeholder(shape=[1], dtype=tf.int32)
state_one_hot = tf.one_hot(indices=state, depth=16)
model = build_model(state_one_hot)
predict = tf.argmax(input=model, axis=1)

expected = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(expected - model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()


# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []


with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs', sess.graph)

    # out = sess.run(model, feed_dict={x: model_inputs})
    # print(out)

    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1

            # Choose an action by greedily (with e chance of random action)
            # from the Q-network
            a, q_values = sess.run([predict, model],
                feed_dict={
                    state: [s]
                })
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            # print(input_state)
            # print(soh)
            # print(a)
            # print(q_values)
            #Get new state and reward from environment
            s1, reward, done, _ = env.step(a[0])
            # Obtain the Q' values by feeding the new state through
            # our network
            Q1 = sess.run(model,
                feed_dict={
                    state: [s1]
                })

            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = q_values
            targetQ[0, a[0]] = reward + y*maxQ1
            # print(targetQ)
            # Train our network using target and predicted Q values
            _ = sess.run([updateModel],
                feed_dict={
                    state: [s],
                    expected: targetQ
                })

            rAll += reward
            s = s1

            # input("Press Enter to continue")
            if done == True:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)

    writer.close()

print("Percent of succesful episodes: {}%".format(
    sum(rList)/num_episodes * 100))
print("Percent of succesful episodes (last 1000): {}%".format(
    sum(rList[-1000:])/1000 * 100))
plt.plot(rList)
plt.show()
plt.plot(jList)
plt.show()

