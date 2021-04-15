from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
imggn = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training = imggn.flow_from_directory('/Users/rishabhjain/Desktop/dataset/training_set', target_size=(64, 64), class_mode='binary', batch_size=32)
test_gen = ImageDataGenerator(rescale=1.0/255)
testing = test_gen.flow_from_directory(directory='/Users/rishabhjain/Desktop/dataset/test_set', target_size=(64, 64), class_mode='binary', batch_size=32)
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training, validation_data=testing,epochs=25)
cnn.save("DogCat")
from keras.models import load_model
cnn = load_model("DogCat")
import numpy as np
from tensorflow.keras.preprocessing import image
test = image.load_img('/Users/rishabhjain/Downloads/123.jpeg', target_size=(64, 64))
test = image.img_to_array(test)
test = np.expand_dims(test, axis=0)
result = cnn.predict(test)
# print(training.class_indices)
if result[0][0] == 1:
    print("Dog")
else:
    print("Cat")
