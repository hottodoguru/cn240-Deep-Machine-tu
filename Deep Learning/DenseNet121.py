import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.applications import DenseNet121
from keras_preprocessing.image import ImageDataGenerator



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.config.list_physical_devices('GPU')


from keras import models
from keras import layers
from keras import optimizers
from keras.models import Model,load_model,Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D,BatchNormalization,Activation

##### Import DenseNet121
d121_conv = DenseNet121(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 1024))
d121_conv.summary()

##### ImageDatagaGenerator
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10
train_features = np.zeros(shape=(nTrain, 7, 7, 1024))
train_labels = np.zeros(shape=(nTrain,3))

train_datagen2 = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    validation_split = 0.2)
model_path = "E://CN240//model//model5-{epoch:02d}-{val_accuracy:.4f}.h5"


train_gen = train_datagen2.flow_from_directory(
directory='C://Users//Admin//Desktop//Newtrain',
target_size=(224, 224),
shuffle = True,
color_mode="rgb",
class_mode="categorical",
subset = 'training')

valid_gen = train_datagen2.flow_from_directory(
directory='C://Users//Admin//Desktop//Newtrain',
target_size=(224, 224),
color_mode="rgb",
class_mode="categorical",
subset='validation')

    
test_gen = train_datagen2.flow_from_directory(
directory='C://Users//Admin//Desktop//testdata',
target_size=(224, 224),
color_mode="rgb",
class_mode='categorical')


##### Train Model

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
generator = train_gen
    
valid = valid_gen
test = test_gen
batch_size = 16


filepath="E://CN240//model_from_fold//dense//modelDense-{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(
                generator,
                steps_per_epoch=generator.n/batch_size,
                epochs=30,
                validation_data=valid,
                validation_steps=valid.n/batch_size,
                shuffle=True,
                verbose=1,
                callbacks = callbacks_list)

import json
from keras.models import model_from_json, load_model
with open('G://Deep Model//Densenet//model5_architecture.json', 'w') as f:
    f.write(model.to_json())
print("Saved model to disk")



##### Graph Accuracy and Loss

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model4 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model4 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


##### Finding Confusion Matrix
ground_truth = validation_generator.classes
label2index = validation_generator.class_indices
label2index

idx2label = dict((v,k) for k,v in label2index.items())
predictions = model.predict_generator(test_features,steps=1, verbose=0)
prob = model.predict(test_features)
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nTest))

import sklearn
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(test_features)

print('\n', sklearn.metrics.classification_report(np.where( test_labels > 0)[1], np.argmax(y_pred, axis=1)))


def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                         title = 'Confusion Matrix',
                         cmap=plt.cm.Blues):
  
  plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation = 45)
  plt.yticks(tick_marks, classes)
  
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
    print("Normalized Confusion Matrix")
  else:
    print("Confusion Matrix without normalization")
  
  print(cm)
  
  thresh = cm.max() / 2.
  
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i, cm[i,j],
            horizontalalignment ="center",
            color = "white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



x_labels = test_labels.argmax(axis=1)
y_labels = predictions.argmax(axis=1)

cm = confusion_matrix(x_labels, y_labels)
cm_plot_labels = ['Glaucoma','Normal','Others']

import itertools

plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')
