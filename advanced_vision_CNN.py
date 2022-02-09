# Unzipping data file
import os
import zipfile
 
local_zip = 'C:\\Users\\JK Kim\\Downloads\\horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:\\Users\\JK Kim\\Downloads\\horse-or-human')
zip_ref.close()

# Labeling is done through directory names, rather than explicit labeling of every image
# Directory with our training horse pictures
train_horse_dir = os.path.join('C:\\Users\\JK Kim\\Downloads\\horse-or-human\\horses')
 
# Directory with our training human pictures
train_human_dir = os.path.join('C:\\Users\\JK Kim\\Downloads\\horse-or-human\\humans')

# Total count of images
# print('total training horse images:', len(os.listdir(train_horse_dir)))
# print('total training human images:', len(os.listdir(train_human_dir)))

# Filenames
train_horse_names = os.listdir(train_horse_dir)
# print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
# print(train_human_names[:10])

# # Displaying some data samples for ourselves
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
 
# # Parameters for our graph; we'll output images in a 4x4 configuration
# nrows = 4
# ncols = 4
 
# # Index for iterating over images
# pic_index = 0

# # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)
 
# pic_index += 8
# next_horse_pix = [os.path.join(train_horse_dir, fname) 
#                 for fname in train_horse_names[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir, fname) 
#                 for fname in train_human_names[pic_index-8:pic_index]]
 
# for i, img_path in enumerate(next_horse_pix + next_human_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)
 
#   img = mpimg.imread(img_path)
#   plt.imshow(img)
 
# plt.show()
 
# CNN Model
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

# Loss function is binary crossentropy because this is a binary classification problem and the final activation function is a sigmoid
# For optimizer function, RMSProp optimization algo is preferable to Stochastic Gradient Descent 
  # because RMSProp automates learning-rate tuning for me 
  # other optimizers like Adam and AdaGrad also automate learning-rate tuning and would work just as well here
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# Set up data generators that read pictures in the source folders, convert them to float32 tensors, and feed them (with labels) to the network
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# There will be generators for both train and test, which yield batches of 300x300 images with labels (0 or 1)
# Normalize the data first (uncommon to feed CNN raw pixels)
# In Keras, that can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter
  # That ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via 
  # .flow(data, labels) or .flow_from_directory(directory)
  # Those generators can then be used with the Keras model methods that accept 
  # data generators as inputs: fit_generator, evaluate_generator and predict_generator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
 
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\JK Kim\\Downloads\\horse-or-human\\',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# 15 epochs, may take a bit
# Note the values per epoch, the loss and accuracy are great indicators of progress of training
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=10,
      verbose=1)

# This model has overfitted because the data set is small (~500 images)
# The more data, the better!
# There are many techniques to make training better even with limited data, such as image augmentation

# Testing the model through custom input, unfortunate that I'm not coding on colab :(
import numpy as np
# from google.colab import files
# from keras.preprocessing import image
 
# uploaded = files.upload()
 
# for fn in uploaded.keys():
 
#   # predicting images
#   path = '/content/' + fn
#   img = image.load_img(path, target_size=(300, 300))
#   x = image.img_to_array(img)
#   x = np.expand_dims(x, axis=0)
 
#   images = np.vstack([x])
#   classes = model.predict(images, batch_size=10)
#   print(classes[0])
#   if classes[0]>0.5:
#     print(fn + " is a human")
#   else:
#     print(fn + " is a horse")

# import numpy as np # already imported
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
 
# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)
 
img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
 
# Rescale by 1/255
x /= 255
 
# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)
 
# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]
 
# Now let's display our representations
# Note: zip is a parallel iteration function
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      if x.std()>0:
        x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()