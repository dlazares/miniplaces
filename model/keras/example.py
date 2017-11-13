import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from load_miniplaces import loadMiniplaces
from keras.utils import to_categorical

# Generate dummy data
# x_train = np.random.random((100, 100, 100, 3))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# x_test = np.random.random((20, 100, 100, 3))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)



train_data_list = '../../data/train.txt'
val_data_list = '../../data/val.txt'
images_root = '../../data/images/'
(x_test, y_test, x_train, y_train) = loadMiniplaces(train_data_list, val_data_list, images_root,num_train=100,num_val=100,size=[100,100])

print(x_train.shape)
print(x_train[0])
x_train = x_train.reshape(-1, 100, 100, 3).astype('float32') / 255.
x_test = x_test.reshape(-1, 100, 100, 3).astype('float32') / 255.
y_train = to_categorical(y_train.astype('float32'),num_classes=100)
y_test = to_categorical(y_test.astype('float32'),num_classes=100)


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mae', 'acc'])

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)
