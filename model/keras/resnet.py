import numpy as np
import os
import argparse
import keras

from keras.models import Sequential, Model
from keras import layers
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.utils.vis_utils import plot_model
from load_miniplaces import loadMiniplaces,loadMiniplacesBatch


def resnet():
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
    return model

def resnet32():
    cardinality = 32
    x = Input(shape=(100, 100, 3))
    y = residual_network(x,cardinality,100)
    return Model(inputs=x,outputs=y)


def residual_network(x,cardinality,output_shape):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y


    # conv1
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(output_shape,activation="softmax")(x)

    return x

def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    # model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #               loss=[margin_loss, 'mse'],
    #               loss_weights=[1., args.lam_recon],
    #               metrics={'out_caps': 'accuracy'})
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mae', 'acc','top_k_categorical_accuracy'])

    
    # Training without data augmentation:
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, callbacks=[log, tb, checkpoint, lr_decay],validation_data=(x_test,y_test))
    

    # # Begin: Training with data augmentation ---------------------------------------------------------------------#
    # def train_generator(x, y, batch_size, shift_fraction=0.):
    #     train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
    #                                        height_shift_range=shift_fraction)  
    #     generator = train_datagen.flow(x, y, batch_size=batch_size)
    #     while 1:
    #         x_batch, y_batch = generator.next()
    #         yield ([x_batch, y_batch], [y_batch, x_batch])

    # # Training with data augmentation. If shift_fraction=0., also no augmentation.
    # model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
    #                     steps_per_epoch=int(y_train.shape[0] / args.batch_size),
    #                     epochs=args.epochs,
    #                     validation_data=[[x_test, y_test], [y_test, x_test]],
    #                     callbacks=[log, tb, checkpoint, lr_decay])
    # # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def trainBatch(model, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    # model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #               loss=[margin_loss, 'mse'],
    #               loss_weights=[1., args.lam_recon],
    #               metrics={'out_caps': 'accuracy'})
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mae', 'acc','top_k_categorical_accuracy'])

    groups = args.groups
    for i in range(groups):
        print("Training Group: ",i)
        (x_test, y_test, x_train, y_train) = loadMiniplacesBatch(train_data_list, val_data_list, images_root,group=i,groups=groups,size=[100,100])
        x_train = x_train.reshape(-1, 100, 100, 3).astype('float32') / 255.
        x_test = x_test.reshape(-1, 100, 100, 3).astype('float32') / 255.
        y_train = to_categorical(y_train.astype('float32'),num_classes=100)
        y_test = to_categorical(y_test.astype('float32'),num_classes=100)
        print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

        # Training without data augmentation:
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, callbacks=[log, tb, checkpoint, lr_decay],validation_data=(x_test,y_test))

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred= model.predict(x_test)
    from keras import backend as K
    import tensorflow as tf
    print("y_pred shape:",y_pred.shape)
    top_values, top_indices = K.get_session().run(tf.nn.top_k(y_pred, k=5))
    print(top_indices.shape)
    top5 = 0
    y = np.argmax(y_test, 1)
    for i in range(len(y)):
        #print(y[i],top_indices[i],y[i] in top_indices[i])
        if y[i] in top_indices[i]:
            top5 += 1
    print('Top 5 acc: ', top5/len(y),top5," out of ",len(y))
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])


if __name__ == "__main__":
    
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--groups', default=1000, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ### Load Data ###
    train_data_list = '../../data/train.txt'
    val_data_list = '../../data/val.txt'
    images_root = '../../data/images/'

    (x_test, y_test, x_train, y_train) = loadMiniplaces(train_data_list, val_data_list, images_root,num_train=100,num_val=10000,size=[100,100])
    x_train = x_train.reshape(-1, 100, 100, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 100, 100, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'),num_classes=100)
    y_test = to_categorical(y_test.astype('float32'),num_classes=100)

    model = resnet32()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        # train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        model.summary()
        trainBatch(model=model,args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        else:
            print("Loading weights from ",args.weights)
        test(model=model, data=(x_test, y_test))
