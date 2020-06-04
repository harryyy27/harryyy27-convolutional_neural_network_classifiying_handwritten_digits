import pandas as pd
import numpy as np
from keras import models
from scipy.stats import mode
from keras import layers
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adam
from keras.datasets import mnist
################################################## LOAD DATA ######################################################
train_data = pd.read_csv('./digit-recognizer/train.csv',delimiter=',')
test_data = pd.read_csv('./digit-recognizer/test.csv',delimiter=',')
# print(len(test_data))
train_labels = train_data['label']
train_data = train_data.drop(['label'],axis=1)
################################################## INSPECT DATA ######################################################

# print(train_data.head())
# print(train_data.shape)
# print(train_labels)
################################################## DETERMINE NULL VALUES ######################################################

# null_values = train_data.isnull().any().describe()

# print(null_values)
"""no null values""" 


################################################## RESHAPE INTO NUMPY ARRAY ######################################################
train_data = train_data.to_numpy().reshape((42000,28,28,1))
test_data = test_data.to_numpy().reshape((28000,28,28,1))
"""turn into 4D array (grayscale bitmap requires additional dimension)"""
# print(train_data)
# print(train_data.shape)

################################################## NORMALIZE ######################################################

train_data = train_data.astype('float32')/255
test_data = test_data.astype('float32')/255
"""turn data into float 32"""

# print(train_data)


################################################## ONE HOT ENCODE LABELS ######################################################
"""Encode target variable is not a must when you use a multiclass classification. It depends on the loss function you use. If you don't want to encode target varaible, you must use sparsecategoricalcrossentropy as loss function. Otherwise, you use categorical_crossentropy as loss function and you have to encode target variable."""

train_labels = to_categorical(train_labels,num_classes=10)
print(train_labels)
################################################## SPLIT INTO VALIDATION + TEST DATA #####################################################

val_data = train_data[:5000]
train_data = train_data[5000:len(train_data)+1]
################################################## DATA AUGMENTATION ######################################################

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(train_data)
val_labels = train_labels[:5000]
train_labels = train_labels[5000:len(train_labels)+1]
print(len(val_data))
print(len(train_data))
print(len(val_labels))
print(len(train_labels))

print(val_data.shape)
print(train_data.shape)

kernel_size_arr=[3,4,5]
ensemble_model=[]
for i in kernel_size_arr:
    model = models.Sequential()
    model.add(layers.Conv2D(filters = 64, kernel_size = (i,i),padding = 'Same',use_bias=False, input_shape = (28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))                 
    model.add(layers.Conv2D(filters = 64, kernel_size = (i,i),padding = 'Same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))


    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10, activation='softmax'))
    # optimizer = RMSProp(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    model.fit_generator(datagen.flow(train_data,train_labels, batch_size=64),
                                epochs = 30, validation_data = (val_data,val_labels),
                                verbose = 2, steps_per_epoch=train_data.shape[0] // 64
                                , callbacks=[learning_rate_reduction])
    # history = model.fit_generator(datagen.flow(train_data,train_labels, batch_size=64),
    #                               epochs = 30, validation_data = (val_data,val_labels),
    #                               verbose = 2, steps_per_epoch=train_data.shape[0] // 64
    #                               , callbacks=[learning_rate_reduction])

    ensemble_model.append(model)
    model.save("model"+str(i)+".h5")
ensemble_model = np.array(ensemble_model)
final_predictions = []
###############################DETERMINE ACCURACY OF ENSEMBLE###############################
for i in range(len(val_data)):
    predictions = []
    for model in ensemble_model:
        predict = model.predict(np.array([val_data[i]]))
        prediction = max(predict[0])
        int_prediction = predict[0].tolist().index(prediction)
        predictions.append(int_prediction)
    most_votes = mode(predictions)
    
    if val_labels[i][most_votes]==1:
        final_predictions.append(True)
    else:
        final_predictions.append(False)

val_acc = final_predictions.count(True) / len(final_predictions)
val_loss = final_predictions.count(False) / len(final_predictions)
print('Validation Accuracy')
print(val_acc)
print('Validation Loss')
print(val_loss)

###############################DETERMINE ACCURACY OF ENSEMBLE###############################
# for i in range(len(test_data)):
#     predictions = []
#     for model in ensemble_model:
#         predict = model.predict(np.array([val_data[i]]))
#         prediction = max(predict[0])
#         int_prediction = predict[0].tolist().index(prediction)
#         predictions.append(int_prediction)
#     try:
#         most_votes = mode(predictions)
#         final_predictions.append(most_votes)
#     except:
#         final_predictions.append(predictions[1])