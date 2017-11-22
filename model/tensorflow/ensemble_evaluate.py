import numpy as np
import pickle
import tensorflow as tf
import os

softmax_files = ["./softmax_results/resnet_finetunev2-5000.p"]

softmax_results = np.zeros(shape=(10002,100))

for fn in softmax_files:
    print("loading from ",fn)
    result = pickle.load( open( fn, "rb" ) )
    print(type(result),type(result[1][0]))
    print(len(result),len(result[1]))
    # print(result.shape)
    for i in range(1,10001):
        print(i,len(result[i]))
        softmax_results[i] = np.add(softmax_results[i],result[i])

x = tf.placeholder(tf.float32, [10002,100])
out_vals,out_indices = tf.nn.top_k(tf.nn.softmax(x),k=5,sorted=True)
with tf.Session() as sess:

    top_values,top_indices = sess.run([out_vals,out_indices], feed_dict={x: softmax_results})
    print(top_indices[0],top_indices[1],top_indices[10000],top_indices[10001])

    test_dir = "../../data/images/test"
    filenames = sorted(os.listdir(test_dir))
    with open("ensemble_results2.txt","w") as result_file:
        for l in range(len(filenames)):
            fn = filenames[l]
            vals = " ".join(map(str,top_indices[l+1]))
            result_file.write("test/"+fn + " "+vals+"\n")



