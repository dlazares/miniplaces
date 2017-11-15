import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
import resnet_model
import scipy.misc

# Dataset Parameters
batch_size = 64 
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 50
step_save = 500
path_save = './checkpoints/resnet_finetune'
start_from = './checkpoints/resnet-1000'

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/test',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }


loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
# logits = alexnet(x, keep_dropout, train_phase)
resnet_size = 18
num_classes = 100
resnet = resnet_model.imagenet_resnet_v2(resnet_size, num_classes)
logits = resnet(x,True)


# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
    
    files = os.listdir(test_dir)
    group_size = int(len(files) / args.groups)
    with open("results.txt","w") as result_file:
        for i in range(args.groups):
            print("Running on group",i)
            test_im = []
            filenames = []
            for j in range(group_size * i,min(len(files),group_size*(i+1))):
                filepath = os.path.join(test_dir, files[j])
                image = scipy.misc.imread(filepath)
                image = scipy.misc.imresize(image, (size[0], size[1],3))
                test_im.append(image)
                filenames.append(files[j])
            test_im = np.array(test_im)
            test_im = test_im.reshape(-1, 100, 100, 3).astype('float32') / 255.
            print("predicting on ",test_im.shape[0]," images")
            out = tf.nn.top_k(tf.softmax(logits),k=5,sorted=True)
            top_values,top_indices = sess.run([out], feed_dict={x: test_im, y: labels_batch, keep_dropout: 1., train_phase: False}) 

            print("writing to file")
            for l in range(len(filenames)):
                fn = filenames[l]
                vals = " ".join(map(str,top_indices[l]))
                result_file.write("test/"+fn + " "+vals+"\n")

   