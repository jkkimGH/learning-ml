import tensorflow as tf

# Callbacks
# Essentially a way to reach a desired accuracy and then stop training
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


# 1. Load MNIST clothing data set
mnist = tf.keras.datasets.fashion_mnist

# 2. Call load_data() to get both training and testing set with images and labels
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Print data for item 0 (label)
# import matplotlib.pyplot as plt
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

# Notice how there are numbers between 0 and 255
# But since we're doing NN, we want to **normalize** them by using just 0 and 1
# Python provides an easy way for us to do that, like that below
# 3. Normalizing
training_images  = training_images / 255.0
test_images = test_images / 255.0

# 4. Defining the NN
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    # 1. Sequential defines a sequence of layers in the neural network.
    # 2. Flatten takes a square and turns it into a one-dimensional vector.
    # 3. Dense adds a layer of neurons.
    # 4. Activation functions tell each layer of neurons what to do. There are lots of options, but use these for now:
        # 1. Relu effectively means that if X is greater than 0 return X, else return 0. 
            # It only passes values of 0 or greater to the next layer in the network.
        # 2. Softmax takes a set of values, and effectively picks the biggest one. 
            # For example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], 
            # then it saves you from having to sort for the largest value—it returns [0,0,0,0,1,0,0,0,0].

# 5. Building the NN
    # 1. Create the NN by compiling it with optimizer and loss functions
    # Here, we use the metrics parameter, which we use to have tensorflow report on its accuracy on training set with its evolving predictions
    # Callbacks is used here too
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # 2. Then train it on the training set
    # Callbacks is used here, but it seems like the model is not able to reach 95% accuracy with 30 epochs (was 5 originally kekw)
model.fit(training_images, training_labels, epochs=30, callbacks=[callbacks])

# The model gets around 89% accurate, which is not bad considering it trained for only 5 epochs

# 6. Test the model!
# 87%, not as accurate as on the training set but we can make it better :)
model.evaluate(test_images, test_labels)

# 7. Exercises!
# Exercise 1: Run this code, and try to figure out what the output it suppose to mean!
# Creating a set of classifications & printing the first entry
# classifications = model.predict(test_images)
# This prints out 10 numbers (because 10 labels) and each number represents the probability of the item being assigned the corresponding label
# These 10 numbers sum up to 1 because of Softmax
# print(classifications[0])
# Printing out the first entry's label
# print(test_labels[0])

# At times, these can be a little inaccurate because our model is only around ~88% accurate

# Exercise 2: Mess around with the number of Dense layers' neurons
# For example, if you increase the number of neurons to 1024, it will slow down the calculations but will yield better results
# However, more is not always better, as you can hit the law of diminishing returns very quickly

# Exercise 3: What if we removed the Flatten layer?
# We would get an error due to the shape of the data
# Rule of thumb: first layer of the NN should be the same shape as your data
# Since the data used in this case is 28x28, we'd have to make the first layer 28 layers of 28 neurons, which seems ridiculous
# So, just flatten it to a 784x1 vector

# Exercise 4: Try messing with the final layer's neuron count
# Error as soon as the NN encounters an unexpected value
# Another rule of thumb: the last layer's neuron count should be the same as the number of labels

# Exercise 5: Try adding another layer between the first and last Dense layers
# In this case, it doesnt really have much of an impact as the data is relatively simple
# However, in more complex cases, extra layers are often necessary

# Exercise 6: Try removing the part where we normalize the data & do you think and why will the output change?
# Since different inputs have different range and therefore scale, you should normalize to give them an equal importance as a feature
# However, normalization does not always increase accuracy:
# You never know until it is implemented
# It depends on at which stage in you training you apply normalization, on whether you apply normalization after every activation, etc.
# But if you do normalize, the calculations tend to be faster