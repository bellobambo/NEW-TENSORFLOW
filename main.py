import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Initializing Tensors
x = tf.constant(4 , shape=(1,1), dtype=tf.float32)
x = tf.constant([[1,2,3], [4,5,6]])
x = tf.ones((2,4))
x = tf.eye(3)
x = tf.random.normal((1,3), mean=0, stddev=1)
x = tf.range(9)


# Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])


z = y - x
z = y * x
z = y + x
z = y / x
z = tf.tensordot(x, y, axes=1)

z = x ** 5

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x,y)
print(z)

z = x @ y
print(z)


# Indexing

x = tf.constant([0,1,1,2,3,1,2,3])
# print(x[:])
# print(x[1:])
# print(x[1:3])
# print(x[::-1])


indices = tf.constant([0,3])
x_ind = tf.gather(x, indices)
x = tf.constant([[1,2],
                 [3,4],
                 [5,6]])

# print(x[0,:])
# print(x[0:2, : ] )

# Reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x,(3,3))
print(x)

x = tf.transpose(x,perm=[1,0])