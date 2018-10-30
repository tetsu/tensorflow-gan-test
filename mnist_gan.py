import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    return inputs_real, inputs_z


def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    """Generate image using whatever image input
    'Generate image using whatever image input'

    Args:
        z (int): Input dimension
        out_dim (int): output dimension
        n_units (int):  number of units
        reuse (boolean): reuse data or not
        alpha (float): Leaky ReLu coefficient

    Returns:
        tuple: output tensor as tuple of (out_dim) dimensions
    """
    with tf.variable_scope('generator', reuse=reuse):
        # first hidden layer using leaky Relu not to make the output of x < 0  to zero.
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)

        # second hidden layer
        # use tanh to transform output between -1 to 1 using Hyperbolic Tangent
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)

        return out


def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    # Check
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits


if __name__ == '__main__':
    # initialize hyper parameters (global variables)
    input_size = 784 # 28 * 28
    z_size = 100
    generator_hidden_size = 128
    discriminator_hidden_size = 128
    alpha = 0.01
    smooth = 0.1

    # define graph
    tf.reset_default_graph()
    input_real, input_z = model_inputs(input_size, z_size)

    generator_model = generator(input_z, input_size, n_units=generator_hidden_size, alpha=alpha)
    discriminator_model_real, discriminator_logits_real = discriminator(input_real, discriminator_hidden_size, False, alpha=alpha)
    discriminator_model_fake, discriminator_logits_fake = discriminator(generator_model, discriminator_hidden_size, True, alpha)

    # define loss functions
    discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_logits_real,
        labels=tf.ones_like(discriminator_logits_real)*(1-smooth)
    ))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_logits_fake,
        labels=tf.zeros_like(discriminator_logits_real)
    ))
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_logits_fake,
        labels=tf.ones_like(discriminator_logits_fake)
    ))

    # define optiomization
    learning_rate = 0.002
    trainable_vars = tf.trainable_variables()
    generator_vars = [var for var in trainable_vars if var.name.startswith('generator')]
    discriminator_vars = [var for var in trainable_vars if var.name.startswith('discriminator')]

    generator_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(generator_loss, var_list=generator_vars)
    discriminator_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(discriminator_loss, var_list=discriminator_vars)

    # training
    batch_size = 100
    batch = mnist.train.next_batch(batch_size)
    epochs = 100
    samples = []
    losses = []
    saver = tf.train.Saver(var_list=generator_vars)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for i in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size, input_size))
                batch_images = batch_images * 2 -1

                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                _ = sess.run(discriminator_train_optimize, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(generator_train_optimize, feed_dict={input_z: batch_z})

            train_loss_discriminator = sess.run(discriminator_loss, {input_real: batch_images, input_z: batch_z})
            train_loss_generator = generator_loss.eval({input_z: batch_z})

            print('Epoch {}/{}'.format(e+1, epochs),
                'D Loss: {:.4f}'.format(train_loss_discriminator),
                'G Loss: {:.4f}'.format(train_loss_generator))

            losses.append({train_loss_discriminator, train_loss_generator})

            sample_z = np.random.uniform(-1, 1, size=(16, z_size))
            gen_samples = sess.run(
                generator(input_z, input_size, n_units=generator_hidden_size, reuse=True, alpha=alpha),
                feed_dict={input_z: sample_z}
            )
            samples.append(gen_samples)
            saver.save(sess, './checkpoints/generator.ckpt')

    with open('training_sample.pkl', 'wb') as f:
        pkl.dump(samples, f)
