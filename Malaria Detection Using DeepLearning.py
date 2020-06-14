#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Malaria Detection using Transfer Learning Technique
from keras.layers import Input, Lambda,Dense,Flatten
from keras.models import Model


# In[8]:


from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 


# In[9]:


# Resize all images to this
IMAGE_SIZE=[224,224]


train_path='C:/Users/USER/Downloads/87153_200743_bundle_archive/cell_images/Train'
test_path='C:/Users/USER/Downloads/87153_200743_bundle_archive/cell_images/Test'

# Add preprocessing layes to the front of VGG
vgg= VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
  
# Don't train existing weights
for layer in vgg.layers:
  layer.trainable=False

#Useful for getting number of classes
folders=glob('C:/Users/USER/Downloads/87153_200743_bundle_archive/cell_images/Train/*')

# Our Layers
x=Flatten()(vgg.output)
prediction=Dense(len(folders),activation='softmax')(x)

# Create a model Object
model= Model(inputs=vgg.input, outputs=prediction)


# In[10]:


# View the structure of the model
model.summary()


# In[ ]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/USER/Downloads/87153_200743_bundle_archive/cell_images/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C:/Users/USER/Downloads/87153_200743_bundle_archive/cell_images/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('model_vgg19.h5')


# In[ ]:




