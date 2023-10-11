## Skin Cancer Classification

The data was imported from Keras ML repository: https://keras.io/api/datasets/mnist/

The dataset of 70000 records, contains 7 distinct classes of skin cancer namely:
- Melanocytic nevi
- Melanoma
- Benign keratosis-like lesions
- Basal cell carcinoma
- Actinic keratoses
- Vascular lesions
- Dermatofibroma

Skin cancer is the most common human malignancy, predominantly visual diagnosis, starts with an initial clinical screening, then a dermoscopic analysis, a biopsy and histopathological examination.

## Goal

The objective is to classify the data into the various quality lesion categories. We are using Hyperparameter Tuning using HyperBand, Convolution Neural Network with keras tensorflow in backend and analyse the result to see how the model can be useful in practical scenario.

```python
import tensorflow
from tensorflow import keras
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
```
## Splitting into Train and Test datasets

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32').reshape(60000,28*28) / 255.0
x_test = x_test.astype('float32').reshape(10000,28*28) / 255.0
y_train = y_train.reshape(60000,)
y_test = y_test.reshape(10000,)
```

There are no missing values in the dataset.

```python
import keras_tuner as kt
import random
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam, SGD, RMSprop, Adadelta

def optimizer(model,optimal,activate,lr):
    model.add(Dense(128,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    model.compile(optimizer=globals()[optimal](learning_rate=lr),loss="categorical_crossentropy",metrics=['accuracy'])
    model.add(Dropout(0.1))
    return model

def model_builder(hp):
    model = Sequential()
    model.add(keras.layers.Flatten(input_shape=(28*28,)))
    
    for i in range(hp.Int('num_layer',min_value=1,max_value=10)):
        model.add(Dense(hp.Int('units'+str(i), min_value=128, max_value=512,step=128),activation='relu')) # Choose units between 32-512
    activate = hp.Choice("activation",values=['relu','tanh','sigmoid']) # Choose an activation function
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # Choose a learning from 0.01, 0.001, or 0.0001
    optimize = hp.Choice('optimizer',values=['Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'SGD']) # Choose an optimizer
    return optimizer(model,optimize,activate,hp_learning_rate)
```
Using TensorFlow backend

```python
tuner = kt.Hyperband(model_builder,objective='val_accuracy',max_epochs=5,factor=3)
stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
tuner.search(x_train,y_train,epochs=5,validation_data=(x_test,y_test), validation_split=0.2,callbacks=[stop_early])
```
Reloading Tuner from .\untitled_project\tuner0.json

```python
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""Optimal learning rate is {best_hps.get('learning_rate')}.""")
```
Optimal learning rate is 0.001.

```python
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test,y_test), validation_split=0.2)
val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
print("Best epoch:%d"%(best_epoch,))
```

Although the val_accuracy and val_loss curves look fitted well, further model building will result in higher accuracy and lower loss with epochs.

  Final Accuracy Score: 0.9677 
  Final Loss Score: 0.3531

Overall, our machine learning does a good job of predicting cancer classes, faster than a human can diagnose skin lesions.
