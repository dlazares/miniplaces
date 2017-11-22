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

# Test Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 50
step_save = 500
path_save = './checkpoints/resnet_finetune'
start_from = './checkpoints/resnet_affinev3-2000'
test_dir = "../../data/images/test"
groups=100

opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/test',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

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
    group_size = int(len(files) / groups)
    with open("results_affine.txt","w") as result_file:
        for i in range(groups):
            print("Running on group",i)
            test_im = []
            filenames = []
            for j in range(group_size * i,min(len(files),group_size*(i+1))):
                filepath = os.path.join(test_dir, files[j])
                image = scipy.misc.imread(filepath)
                image = scipy.misc.imresize(image, (load_size, load_size))
                image = image.astype(np.float32)/255.
                image = image - np.array(data_mean)
            
                offset_h = (load_size-fine_size)/2
                offset_w = (load_size-fine_size)/2

                offset_h = int(offset_h)
                offset_w = int(offset_w)    
            
                image =  image[offset_h:offset_h+fine_size, offset_w:offset_w+fine_size, :]

                test_im.append(image)
                filenames.append(files[j])
                
            test_im = np.array(test_im)
            
            print("predicting on ",test_im.shape[0]," images")
            out_vals,out_indices = tf.nn.top_k(tf.nn.softmax(logits),k=5,sorted=True)
            top_values,top_indices = sess.run([out_vals,out_indices], feed_dict={x: test_im,  keep_dropout: 1., train_phase: False}) 

            print("writing to file",filenames[0],top_indices[0],top_values[0])
            for l in range(len(filenames)):
                fn = filenames[l]
                vals = " ".join(map(str,top_indices[l]))
                result_file.write("test/"+fn + " "+vals+"\n")

   
