import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
from game2048.displays import Display, IPythonDisplay
import numpy as np

display1 = Display()
display2 = IPythonDisplay()
inputs = (4, 4, 16)

model=Sequential()
model.add(Conv2D(filters= 128, kernel_size=(4,1),kernel_initializer='he_uniform', padding='Same', activation='relu',input_shape=inputs)) 
model.add(Conv2D(filters= 128, kernel_size=(1,4),kernel_initializer='he_uniform', padding='Same', activation='relu'))
model.add(Conv2D(filters= 128, kernel_size=(2,2),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(3,3),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(4,4),kernel_initializer='he_uniform', padding='Same', activation='relu'))

model.add(Flatten()) 
model.add(BatchNormalization())

for width in [512,128]:
    model.add(Dense(width, kernel_initializer='he_uniform',activation='relu'))
model.add(Dense(4, activation='softmax'))  
model.summary()
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])
model.save('2048model.h5')
