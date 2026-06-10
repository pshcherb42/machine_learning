import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# prepare data 

# define tuples and get daata from a dataset
# dataset returns arrays of pixels
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# scale it down. All values are between 0 and 1
training_images, testing_images = training_images / 255, testing_images / 255

# define class name list 
# visualize 16 images from the dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# grid 4x4 and we iterate through it
# we're getting the label(number) of particular image
# we're passing this number as the index for our class list
'''
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
'''
# plt.show()


# training the model and saving the results inside a file
'''
# building and training the model
# reduce the amount of images that we training the model with
# to safe resurces and time
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# define the neuronal network
model = models.Sequential()
# 32 neurons and 3x3 convolution matrix,  
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# max pooling layer
# each time I have a convolutional layer I have a maxpooling layer to simplify the results
# and reduce it to the essential information
model.add(layers.MaxPooling2D((2, 2)))
# add another convolutional layer
# it looks what we have on an image
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# flatten the input, make it one-dimensional
# (example)make it into a straight layer of 100 units
model.add(layers.Flatten())
# dense layers
model.add(layers.Dense(64, activation='relu'))
# final layer, scales all results to one, so I know how probably one particular classification is the one we are looking for
model.add(layers.Dense(10, activation='softmax'))

# convolutional layer filters for the features in image
# like pointy ears on a cat or long legs of a horse
# maxpooling layer reduces the image to essential information
# then we put one more layer of complexity between flat layer and final result
# lastly we scale the results so we have percentage for existing classifications


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# we want to train the model just once so we're going to save
# training results 

loss, accuracy = model.evaluate(testing_images, testing_labels)
# loss indicates how wrong our model is
# accuracy indicates how many of the examples were classified correctly
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')
'''
# loss 1.94
# accuracy 53%
# not great results but good enough
# if we train it with more examples this variables will get better
# but we would need a more powerfull computer
# we trained it just with 20000 examples
# its a pretty good result for a machine that doesnt know
# what a horse is anyways

# load the model that we just trained
model = models.load_model('image_classifier.keras')

# now we can download the images
# scale them to 32x32 pixels and upload them to the directory

# we are going to use opencv and numpy to load the images
# into the script
# make the predictions
# and visualize the prediction

# load the image
img = cv.imread('car.jpg')
if img is None:
    print("Error: image not found, check the path!")
else:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# bgr to rgb(opencv loads bgr colors(inverted blue and red)) why?
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# show the image
plt.imshow(img, cmap=plt.cm.binary)

# add a prediction
# in the beningin we divided all the images by 255 we have to do it here too
prediction = model.predict(np.array([img]) / 255)
# index of prediction
# highest activation neuron
index = np.argmax(prediction)
# we also need index for class name
print(f'Prediction is {class_names[index]}')
plt.show()