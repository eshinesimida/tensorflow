# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:13:49 2018

@author: Administrator
"""

import os, shutil

original_dataset_dir = '/Users/Administrator/train'
base_dir = '/Users/Administrator/cats_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(base_dir, 'validation')
os.mkdir(val_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

val_cats_dir = os.path.join(val_dir, 'cats')
os.mkdir(val_cats_dir)

val_dogs_dir = os.path.join(val_dir, 'dogs')
os.mkdir(val_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(val_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(val_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu',
          input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()


#model.add(layers.MaxPooling2D(128, (3,3), activation = 'relu'))

from keras import optimizers

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 1e-4),
              metrics = ['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(1./255)
val_datagen = ImageDataGenerator(1./255)
test_datagen = ImageDataGenerator(1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = 'binary')

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = 'binary')

#for data_batch, labels_batch in train_generator:
#    print(data_batch.shape)

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 5,
        validation_data = val_generator,
        validation_steps = 50)

model.save('cats_and_dogs_small_1.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,  len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
history = model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)


###############################

#############################
############################
import numpy as np
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(val_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

from keras.layers import Activation, Dropout, Flatten, Dense
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential, Model
#model = models.Sequential()





model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape = (4 , 4 , 512)))
#model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, 
                    train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))










import os
import numpy as np
from keras.models import Sequential, Model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.utils.np_utils import to_categorical
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
imgs = []
labels = []
img_shape =(150,150)
# image generator
files = os.listdir(test_dir)
# read 1000 files for the generator
for img_file in files[:1000]:
    img = imread('data/test/' + img_file).astype('float32')
    img = imresize(img, img_shape)
    imgs.append(img)

imgs = np.array(imgs)
train_gen = ImageDataGenerator(
     # rescale = 1./255,
     featurewise_center=True,
     featurewise_std_normalization=True,
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True)
val_gen = ImageDataGenerator(
     # rescale = 1./255,
     featurewise_center=True,
     featurewise_std_normalization=True)

train_gen.fit(imgs)
val_gen.fit(imgs)

# 4500 training images 
train_iter = train_gen.flow_from_directory('data/train',class_mode='binary',
                                            target_size=img_shape,   batch_size=16)
# 501 validation images
val_iter = val_gen.flow_from_directory('data/val', class_mode='binary', 
                                        target_size=img_shape, batch_size=16)

'''
# image generator debug
for x_batch, y_batch in img_iter:
    print(x_batch.shape)
    print(y_batch.shape)
    plt.imshow(x_batch[0])
    plt.show()
'''

# finetune from the base model VGG16
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
base_model.summary()

out = base_model.layers[-1].output
out = layers.Flatten()(out)
out  = layers.Dense(1024, activation='relu')(out)
# 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
out = layers.Dropout(0.5)(out)
out = layers.Dense(512, activation='relu')(out)
out = layers.Dropout(0.3)(out)
out = layers.Dense(1, activation='sigmoid')(out)
tuneModel = Model(inputs=base_model.input, outputs = out)
for layer in tuneModel.layers[:19]: # freeze the base model only use it as feature extractors
    layer.trainable = False
tuneModel.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc'])

history = tuneModel.fit_generator(
        generator=train_iter,
        steps_per_epoch=100,
        epochs=100,
        validation_data=val_iter,
        validation_steps=32
        )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.legend()
plt.show()


##################################
import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils.np_utils import to_categorical
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

imgs = []
labels = []
img_shape = (150, 150)
files = os.listdir('data1/test1')
for img_file in files[:1000]:
    img = imread('data1/test1/' + img_file).astype('float32')
    img = imresize(img, img_shape)
    imgs.append(img)
#plt.plot(img)
imgs = np.array(imgs)
#imgs = np.array(imgs)
train_gen = ImageDataGenerator(
     rescale = 1./255,
     featurewise_center=True,
     featurewise_std_normalization=True,
     rotation_range=40,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True,
     fill_mode = 'nearest')
val_gen = ImageDataGenerator(
     # rescale = 1./255,
     featurewise_center=True,
     featurewise_std_normalization=True)

train_gen.fit(imgs)
val_gen.fit(imgs)

# 4500 training images 
train_iter = train_gen.flow_from_directory('cats_dogs_small/train',class_mode='binary',
                                            target_size=img_shape,   batch_size=16)
# 501 validation images
val_iter = val_gen.flow_from_directory('cats_dogs_small/validation', class_mode='binary', 
                                        target_size=img_shape, batch_size=16)
# define the navie model using sequential model
model = Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
# define the optimzers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc'])
history = model.fit_generator(
        generator=train_iter,
        steps_per_epoch=100,
        epochs=10,
        validation_data=val_iter,
        validation_steps=32
        )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.legend()
plt.show()



######################
# finetune from the base model VGG16
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
base_model.summary()

out = base_model.layers[-1].output
out = layers.Flatten()(out)
out  = layers.Dense(1024, activation='relu')(out)
# 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
out = layers.Dropout(0.5)(out)
out = layers.Dense(512, activation='relu')(out)
out = layers.Dropout(0.3)(out)
out = layers.Dense(1, activation='sigmoid')(out)
tuneModel = Model(inputs=base_model.input, outputs = out)
for layer in tuneModel.layers[:19]: # freeze the base model only use it as feature extractors
    layer.trainable = False
tuneModel.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc'])

history = tuneModel.fit_generator(
        generator=train_iter,
        steps_per_epoch=100,
        epochs=100,
        validation_data=val_iter,
        validation_steps=32
        )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.legend()
plt.show()






