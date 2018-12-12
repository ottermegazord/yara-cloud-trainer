#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

# ## Import libraries and packages

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Add TF GPU for Jetson TX2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

#%matplotlib inline

# tf.__version__, tf.test.is_gpu_available(), tf.test.is_built_with_cuda(), tf.test.gpu_device_name()


# ## Import data set

# In[2]:


data_dir = 'data'
# print(os.listdir(os.path.join(data_dir, 'cirrocumulus', '*.jpg')))


# In[3]:


image_dir = os.path.join(data_dir, 'cirrocumulus', '*.jpg')
print(image_dir)


# In[4]:


def _img_string_to_tensor(image_string, image_size=(299, 299)):
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    
    return image_resized

def make_input_fn(file_pattern, image_size=(32, 32), shuffle=False, batch_size=64, num_epochs=None, buffer_size=4096):
    
    def _path_to_img(path):
        # Get the parent folder of this file to get it's class name
        label = tf.string_split([path], delimiter='/').values[-2]
        print(path)
        # Read in the image from disk
        image_string = tf.read_file(path)
        image_resized = _img_string_to_tensor(image_string, image_size)
        
        return image_resized, label
    
    def _input_fn():
        
        dataset = tf.data.Dataset.list_files(file_pattern)

        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
        else:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.map(_path_to_img, num_parallel_calls=os.cpu_count())
        dataset = dataset.batch(batch_size).prefetch(buffer_size)

        return dataset
    
    dataset = _input_fn()

    return dataset


# ## Define Parameters

# ### Define training parameters

# In[5]:


num_steps = 10000
batch_size = 128


# ### Create dataset

# In[6]:


dataset = make_input_fn(image_dir, image_size=(40,40), batch_size=batch_size, shuffle=True)
print(dataset)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


# ### Define DCGAN network parameters

# In[7]:


# dim_image = 2352 # 28 * 28 * 3
# gen_hidden_dim = 256
# dis_hidden_dim = 256
n_noise = 100


# ## Define DCGAN functions

# ### Define discriminator

# In[8]:

def gen(z, reuse=False):
    momentum = 0.9
    with tf.variable_scope("generator", reuse=None):
        x = z

        d1 = 4  # 3
        d2 = 3

        '''Layer'''
        x = tf.layers.dense(x, units=d1 * d1 * d2)
        x = tf.nn.leaky_relu(x)
        # x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=True, decay=momentum)

        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[10, 10])
        print(x.shape)

        '''Layer'''
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same')
        x = tf.nn.leaky_relu(x)
        # x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=True, decay=momentum)
        print(x.shape)

        '''Layer'''
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same')
        x = tf.nn.leaky_relu(x)
        # x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=True, decay=momentum)
        print(x.shape)

        '''Layer'''
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same')
        x = tf.nn.leaky_relu(x)
        # x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=True, decay=momentum)
        print(x.shape)

        '''Layer'''
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=1, padding='same', activation=tf.nn.sigmoid)
        print(x.shape)
        return x



"""Inv

def gen(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
#         # TensorFlow Layers automatically create variables and calculate their
#         # shape, based on the input.
#         x = tf.layers.dense(x, units=6 * 6 * 128)
#         x = tf.nn.tanh(x)
#         # Reshape to a 4-D array of images: (batch, height, width, channels)
#         # New shape: (batch, 6, 6, 128)
#         x = tf.reshape(x, shape=[-1, 6, 6, 128])
#         # Deconvolution, image shape: (batch, 14, 14, 64)
#         x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
#         # Deconvolution, image shape: (batch, 28, 28, 1)
#         x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
#         # Apply sigmoid to clip values between 0 and 1
#         x = tf.nn.sigmoid(x)

        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        
        '''Layer 1'''
        x = tf.layers.dense(x, units=4*4*1028)
        x = tf.reshape(x, shape=[-1, 4, 4, 1028])
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        '''Layer 2'''
        x = tf.layers.conv2d_transpose(x, 512, 2, strides=2)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        '''Layer 3'''
        x = tf.layers.conv2d_transpose(x, 256, 2, strides=2)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        '''Layer 4'''
        x = tf.layers.conv2d_transpose(x, 128, 2, strides=2)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        '''Layer 5'''
        x = tf.layers.conv2d_transpose(x, 64, 2, strides=2)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        '''Layer 6'''
        x = tf.layers.conv2d_transpose(x, 3, 2, strides=2)
        print(x.shape)
        x = tf.nn.tanh(x)
        return x

        
"""

# ### Define discriminator

# In[9]:

def dis(img_in, reuse=None):
    with tf.variable_scope("discriminator", reuse=reuse):
        '''Layer'''
        x = tf.reshape(img_in, shape=[-1, 40, 40, 3])
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same')
        x = tf.nn.leaky_relu(x)
        # x = tf.layers.dropout(x, keep_prob)
        '''Layer'''
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=1, padding='same')
        x = tf.nn.leaky_relu(x)
        # x = tf.layers.dropout(x, keep_prob)
        '''Layer'''
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same')
        x = tf.nn.leaky_relu(x)
        # x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        '''Layer'''
        x = tf.layers.dense(x, units=128)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x

# ## Assemble GAN

# ### Create placeholders for inputs to GAN

# In[10]:


# input_noise = tf.placeholder(tf.float32, shape=[None, n_noise])
real_input_image = tf.placeholder(tf.float32, shape=[None, 40, 40, 3])
input_noise = tf.placeholder(tf.float32, shape=[None, n_noise])


# ### Create generator network

# In[11]:


generator = gen(input_noise, reuse=False)


# ### Create discriminator networks for fake and real images

# In[12]:


real_dis = dis(real_input_image) # Discriminator for real images
fake_dis = dis(generator, reuse=True) # Discriminator for generated samples

# Concatenate both together
dis_concat = tf.concat([real_dis, fake_dis], axis=0)


# In[13]:


# Stack generator on discriminator
stacked_gan = dis(generator, reuse=True)


# ### Build targets

# In[14]:


dis_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])


# ### Define and build loss function

# In[15]:


dis_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dis_concat, labels=dis_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=stacked_gan, labels=gen_target))


# ### Define separate training variables for each optimizer

# In[16]:


# Generator network variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Define optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002)
optimizer_dis = tf.train.AdamOptimizer(learning_rate=0.002)

# Create training variables
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_dis = optimizer_gen.minimize(dis_loss, var_list=dis_vars)

# initialize variables
init = tf.global_variables_initializer()


# ## Create dataset

# ## Start training

# In[ ]:



# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    sess.run(iterator.initializer)
    for i in range(1, num_steps+1):

        # Prepare Input Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = sess.run(next_element)
        batch_x = np.reshape(batch_x, newshape=[-1, 40, 40, 3])
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, n_noise])

        # Prepare Targets (Real image: 1, Fake image: 0)
        # The first half of data fed to the generator are real images,
        # the other half are fake images (coming from the generator).
        batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
        batch_gen_y = np.ones([batch_size])

        # Training
        feed_dict = {real_input_image: batch_x, input_noise: z,
                     dis_target: batch_disc_y, gen_target: batch_gen_y}
        
#         print("step: %i" %i)
        
        _, _, gl, dl = sess.run([train_gen, train_dis, gen_loss, dis_loss],
                                feed_dict=feed_dict)
#         _, _, gl, dl = sess.run([train_gen, train_dis, gen_loss, dis_loss],
#                                 feed_dict=feed_dict)
        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, n_noise])
        g = sess.run(generator, feed_dict={input_noise: z})
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 1, axis=2),
                             newshape=(40, 40, 3))
            a[j][i].imshow(img)

    f.show()
    plt.show()


# In[ ]:





# In[ ]:




