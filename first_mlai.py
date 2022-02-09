import tensorflow as tf
import numpy as np
from tensorflow import keras
# Very simple machine learning model to correctly guess the function y = 3x + 1

# Create a model with 1 layer and 1 neuron/layer and input shape of 1
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Use stochastic gradient descent as optimizer function and mean squared error as loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Create arrays containing the training data set
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Training the NN, 50 cycles (or epochs)
model.fit(xs, ys, epochs=50)

# Printing a prediction from the model given the data point
print(model.predict([10.0]))