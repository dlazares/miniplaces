import os
import numpy as np
import random
import scipy.misc

# Returns (X_Test, Y_Test, X_Train, Y_train)
# X_Test and X_train are arrays of x by x images
# Y_Test and Y_train are the subsequent labels
def loadMiniplaces(train_data_list, val_data_list, images_root, num_train=100, num_val=100, size=[28, 28]):
    # read data info from lists
    train_im = {} 

    with open(train_data_list, 'r') as f:
        for line in f:
            path, lab = line.rstrip().split(' ')
            lab = int(lab)
            if lab not in train_im:
                train_im[lab] = []
            train_im[lab].append(os.path.join(images_root, path))

    val_im = {} 
    with open(val_data_list, 'r') as f:
        for line in f:
            path, lab = line.rstrip().split(' ')
            lab = int(lab)
            if lab not in val_im:
                val_im[lab] = []
            val_im[lab].append(os.path.join(images_root, path))
    
    numTrainPerClass = int(num_train/ len(train_im))
    numValPerClass = int(num_val / len(val_im))

    X_Train = np.zeros((num_train, size[0], size[1], 3)) 
    Y_Train = np.zeros(num_train)
    
    j = 0 
    for category in train_im:
        for i in range(min(len(train_im[category]),numTrainPerClass)):
            image = scipy.misc.imread(train_im[category][i])
            image = scipy.misc.imresize(image, (size[0], size[1],3))
            X_Train[j] = image.reshape(-1,size[0],size[1],3)
            Y_Train[j] = int(category) 
            j+=1

    X_Test = np.zeros((num_val, size[0], size[1], 3)) 
    Y_Test = np.zeros(num_val)
    j = 0
    for category in val_im:
        for i in range(min(len(val_im[category]),numValPerClass)):
            image = scipy.misc.imread(val_im[category][i])
            image = scipy.misc.imresize(image, (size[0], size[1],3))
            X_Test[j] = image.reshape(-1,size[0],size[1],3)
            Y_Test[j] = int(category) 
            j+=1
    return (X_Test, Y_Test, X_Train, Y_Train)

def loadMiniplacesBatch(train_data_list, val_data_list, images_root, group=0,groups=10, size=[28, 28]):
    train_im = []
    with open(train_data_list, 'r') as f:
        for line in f:
            path, lab = line.rstrip().split(' ')
            lab = int(lab) 
            train_im.append((os.path.join(images_root, path),lab))

    val_im = []
    with open(train_data_list, 'r') as f:
        for line in f:
            path, lab = line.rstrip().split(' ')
            lab = int(lab) 
            val_im.append((os.path.join(images_root, path),lab))

    train_group_size = int(len(train_im)/groups)
    val_group_size = int(len(val_im)/groups)

    x_train = []
    y_train = []
    for i in range(train_group_size*group, min(len(train_im),train_group_size*(group+1))):
        image = scipy.misc.imread(train_im[i][0])
        image = scipy.misc.imresize(image, (size[0], size[1],3))
        x_train.append(image)
        y_train.append(train_im[i][1])

    x_test = []
    y_test = []
    for i in range(val_group_size*group, min(len(val_im),val_group_size*(group+1))):
        image = scipy.misc.imread(val_im[i][0])
        image = scipy.misc.imresize(image, (size[0], size[1],3))
        x_test.append(image)
        y_test.append(val_im[i][1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_test,y_test,x_train,y_train)



#train_data_list = '../../../data/train.txt'
#val_data_list = '../../../data/val.txt'
#images_root = '../../../data/images/'
#x_test,y_test,x_train,y_train = loadMiniplaces(train_data_list, val_data_list, images_root)
#count = {}
#for i in y_test:
#    if i in count:
#        count[i] += 1
#    else:
#        count[i] = 1
#print("count",count)