# CNN: DNN w Convolution layers before Dense layers

import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# The first convolution expects a single tensor containing everything
# So instead of 60000 28x28x1 images in a list, create a 4D list: 60000x28x28x1
# Or error as the convolutions will not recognize the shape
# Reshape param: # of img, x_shape, y_shape, # of color channels
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images / 255.0

model = tf.keras.models.Sequential([
  # Conv2D param: # of convolutions/filters (output_channels), kernal size, activation func, input data shape
  # input_shape param: x_shape, y_shape, # of color channels  
  # Param # for Keras Conv2D = output_channels * (input_channels * kernel_h * kernel_w + 1) (1 for bias)
  # Output shape: output_x_shape, output_y_shape, output_channels
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  # Max Pooling 2x2 halves both x_shape and y_shape while maintaining key features
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(training_images, training_labels, epochs=5)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy * 100))

# Labels for first 100 images, indices 0, 23, 28 are all shoes (label 9)
print(test_labels[:100])

import matplotlib.pyplot as plt
# subplots: create a figure and a set of subplots; param: nrows, ncols, ...
# ax is a singular subplot's canvas
figure, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28
CONVOLUTION_NUMBER = 6

from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs = model.input, outputs = layer_outputs) # Don't need tf.keras. in front?

# ??? no worky... look below
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
# I figured this out! :D
plt.show()

# Exercises:
# From observation, I think the Convolutions highlight a certain part of the shoes, but not always the sole
  # 64 Convolutions resulted in slowest training time, highest accuracy with minimal differences in those of test and train, 100th place decimal
  # 32 Convolutions resulted in lower accuracy and training time
  # 16 Convolutions resulted in worst accuracy and training time, minimal accuracy differences between test and train, 10th place decimal
# Removing the last Convolution resulted in similar accuracy but less training time (64 convolutions)
  # Also removing the last Max Pooling resulted in longer training time, higher accuracies overall, but had ~3% differences in test and train
# Adding another layer of Convolution + Max Pooling made training time highest as 64 2 layer but lower accuracies overall