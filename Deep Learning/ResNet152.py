import tensorflow as tf
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.models import Sequential,Model,load_model
from tensorflow.python.client import device_lib
import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.applications import ResNet152
from sklearn.metrics import classification_report
import itertools
from keras_preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Model,load_model,Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D,BatchNormalization,Activation



##### Pre Processing
from keras_preprocessing.image import ImageDataGenerator
train_gendata = []
valid_gendata = []
test_gendata = []
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
batch_size=4,
class_mode="categorical",
subset = 'training')
train_gendata.append(train_gen)
    
valid_gen = train_datagen2.flow_from_directory(
directory='C://Users//Admin//Desktop//Newtrain',
target_size=(224, 224),
color_mode="rgb",
batch_size=4,
class_mode="categorical",
subset='validation')
valid_gendata.append(valid_gen)
    
test_gen = train_datagen2.flow_from_directory(
directory='C://Users//Admin//Desktop//testdata',
target_size=(224, 224),
color_mode="rgb",
batch_size=1,
class_mode=None)
test_gendata.append(test_gen)


##### Create ResNet152 Model

model = ResNet152(weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3))

x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.55)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.55)(x)
predictions = Dense(3, activation= 'softmax')(x)
model_2 = Model(inputs = model.input, outputs = predictions)
model_2.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


##### Train Model

batch_size = 16
    
generator = train_gendata[0]
    
valid = valid_gendata[0]
test = test_gendata[0]
filepath="E://CN240//model_from_fold//model5-{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor= 'val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
history = model_2.fit(
                generator,
                steps_per_epoch=generator.n/batch_size,
                epochs=30,
                validation_data=valid,
                validation_steps=valid.n/batch_size,
                shuffle=True,
                verbose=1,
                callbacks = callbacks_list)
    
model_2.evaluate_generator(generator=valid,steps=valid.n)
test_gendata[0].reset()
    
import json
from keras.models import model_from_json, load_model
with open('E://CN240//model//model5_architecture.json', 'w') as f:
    f.write(model_2.to_json())
print("Saved model to disk")


##### Ploting Model-Loss Graph

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


##### Printing Report
x_labels = test.classes
y_labels = predicted_class_indices
print(classification_report(y_labels, x_labels))


##### Ploting Confusion Matrix
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

cm = confusion_matrix(x_labels, y_labels)
cm_plot_labels = ['Glaucoma','Normal','Others']
plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')


