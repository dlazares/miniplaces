import os
import numpy as np
import random
import scipy.misc
import matplotlib.pyplot as plt


# Returns (X_Test, Y_Test, X_Train, Y_train)
# X_Test and X_train are arrays of x by x images
# Y_Test and Y_train are the subsequent labels


def loadMiniplaces(train_data_list, val_data_list, images_root, num_train=100, num_val=100, size=[28, 28]):
    # read data info from lists
    train_im = []

    with open(train_data_list, 'r') as f:
        for line in f:
            path, lab = line.rstrip().split(' ')
            train_im.append((os.path.join(images_root, path), int(lab)))

    val_im = []
    with open(val_data_list, 'r') as f:
        for line in f:
            path, lab = line.rstrip().split(' ')
            val_im.append((os.path.join(images_root, path), int(lab)))

    random.shuffle(train_im)
    train_im = train_im[:num_train + 1]

    random.shuffle(val_im)
    val_im = val_im[:num_val + 1]

    X_Train = np.zeros((num_train, size[0], size[1], 3)) 
    Y_Train = np.zeros(num_train)
    for i in range(min(num_train,len(train_im))):
        # image = scipy.misc.imread(train_im[i][0],mode="L")
        image = scipy.misc.imread(train_im[i][0])
        image = scipy.misc.imresize(image, (size[0], size[1],3))
        # print(image.shape)
        # print(image)
        # plt.imshow(np.uint8(image))
        # plt.show()
        # image = image.astype(np.float32)/255.
        X_Train[i] = image.reshape(-1,28,28,3)
        Y_Train[i] = train_im[i][1]

    X_Test = np.zeros((num_val, size[0], size[1], 3)) 
    Y_Test = np.zeros(num_val)
    for i in range(min(num_val,len(val_im))):
        image = scipy.misc.imread(val_im[i][0])
        image = scipy.misc.imresize(image, (size[0], size[1],3))
        # plt.imshow(np.uint8(image))
        # plt.show()
        # image = image.astype(np.float32)/255.
        # np.append(X_Test,image)
        # np.append(Y_Test,val_im[i][1])
        X_Test[i] = image.reshape(-1,28,28,3)
        Y_Train[i] = val_im[i][1]

    # print(X_Train[0].shape)
    # print(X_Train[0])
    # plt.imshow(np.uint8(X_Train[0]))
    # plt.show()
    print(X_Train.shape)
    print(Y_Train.shape)
    print(X_Test.shape)
    print(Y_Test.shape)

    return (X_Test, Y_Test, X_Train, Y_Train)

# train_data_list = '../../../data/train.txt'
# val_data_list = '../../../data/val.txt'
# images_root = '../../../data/images/'
# loadMiniplaces(train_data_list, val_data_list, images_root)