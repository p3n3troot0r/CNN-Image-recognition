import tensorflow as tf

# 0) Data Setup
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 1) Weight Initialization

# initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# using ReLU neurons, so we should initialize them with a slightly positive initial bias to avoid "dead neurons"
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 2) Convolution and Pooling

# Use vanilla convolutions with a stride of one and zero padding to make output size same as input size
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 3) First Convolutional Layer - conv1 + pool1

# The convolutional will compute 32 features for each 5x5 patch

# Its weight tensor will have a shape of [5, 5, 1, 32]
# (The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels.)
W_conv1 = weight_variable([5, 5, 1, 32])
# We will also have a bias vector with a component for each output channel.
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor
# (The second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 4) Second Convolutional Layer - conv2 + pool2

# The second layer will have 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Use outputs from first layer as inputs this time
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 5) Densely Connected Layer - fc1

# Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 6) Dropout

# To reduce overfitting, we will apply dropout before the readout layer.
# We create a placeholder for the probability that a neuron's output is kept during dropout.
# This allows us to turn dropout on during training, and turn it off during testing.
# (TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 7) Readout Layer - Using Softmax Regression (same as other code)

# 10 outputs for each class
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 8) Train and Evaluate the Model
# To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network before. With some additions:
#   - we will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer (a)
#   - we will include the additional parameter keep_prob in feed_dict to control the dropout rate (b)
#   - we will add logging to every 100th iteration in the training process. (c)

# set cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# minimize cost function using Adam optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   # (a)
# generate prediction vector
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# generate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# initialize session variables
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:    # (c)
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})   # (b)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
