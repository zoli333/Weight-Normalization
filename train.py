
import numpy as np
import tensorflow as tf
from model import model_spec
import utils

def convert_to_boolean(flag):
    if flag==1:
        return True
    elif flag==0:
        return False
    

def print_out_all_variables_AND_all_trainable_variables(sess):
    all_variables = tf.all_variables()
    all_variables_vars = sess.run(all_variables)
    for var, val in zip(all_variables, all_variables_vars):
        print(var.name) 
    
    print "\n"
    
    trainable_variables = tf.trainable_variables()
    trainable_variables_vars = sess.run(tf.trainable_variables())
    for var, val in zip(trainable_variables, trainable_variables_vars):
        print(var.name) 
    

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('seed', 3213123, "Seed value for reproductibility")
tf.app.flags.DEFINE_integer('init_batch_size', 500, "Number of examples for initialization")
tf.app.flags.DEFINE_integer('batch_size', 100, "Batch size for training")
tf.app.flags.DEFINE_integer('num_epochs', 200, "Number of epochs for training")

tf.app.flags.DEFINE_integer('use_weight_normalization', 1, "if weightnorm is used")
tf.app.flags.DEFINE_integer('use_batch_normalization', 0, "if batchnorm is used")
tf.app.flags.DEFINE_integer('use_mean_only_batch_normalization', 1, "if mean only batch norm is used")
tf.app.flags.DEFINE_integer('print_out_variables', 0, "Whether to print out trainable and other variables, for debugging")


seed = FLAGS.seed
init_batch_size = FLAGS.init_batch_size
batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs

use_weight_normalization = convert_to_boolean(FLAGS.use_weight_normalization)
use_batch_normalization = convert_to_boolean(FLAGS.use_batch_normalization)
use_mean_only_batch_normalization = convert_to_boolean(FLAGS.use_mean_only_batch_normalization)
print_out_variables = FLAGS.print_out_variables

print 'Weight-normalization used: ' + str(use_weight_normalization)
print 'Batch-normalization used: ' + str(use_batch_normalization)
print 'Mean-only-batch-normalization used: ' + str(use_mean_only_batch_normalization)


if (use_weight_normalization is True and use_batch_normalization is True) or (use_batch_normalization is True and use_mean_only_batch_normalization is True):
    print "Cannot use both!"
    exit (0)




rng = np.random.RandomState(seed)
tf.set_random_seed(seed)

trainx_white, testx_white, trainy, testy = utils.load_data()
trainx_white = np.transpose(trainx_white,(0,2,3,1))  # (N,3,32,32) -> (N,32,32,3)
testx_white = np.transpose(testx_white,(0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
trainy=trainy.astype(np.int32)
testy=testy.astype(np.int32)


nr_batches_train = int(trainx_white.shape[0]/batch_size)
nr_batches_test = int(testx_white.shape[0]/batch_size)


model = tf.make_template('model',model_spec)


x = tf.placeholder(tf.float32,shape=[batch_size,32,32,3])
y = tf.placeholder(tf.int32,shape=[batch_size])
x_init = tf.placeholder(tf.float32,shape=[init_batch_size,32,32,3])



init_forward = model(x_init,keep_prob=0.5,deterministic=False, init=True, 
                        use_weight_normalization=use_weight_normalization,
                        use_batch_normalization=use_batch_normalization, 
                        use_mean_only_batch_normalization=use_mean_only_batch_normalization) # initialization phase


out = model(x,keep_prob=0.5,deterministic=False,init=False,
                        use_weight_normalization=use_weight_normalization,
                        use_batch_normalization=use_batch_normalization, 
                        use_mean_only_batch_normalization=use_mean_only_batch_normalization) # training phase
                        
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=out, name='cross_entropy'))



test_out=model(x,keep_prob=0.5,deterministic=True,init=False,
                        use_weight_normalization=use_weight_normalization,
                        use_batch_normalization=use_batch_normalization, 
                        use_mean_only_batch_normalization=use_mean_only_batch_normalization) # testing phase
                        

argmax=tf.argmax(test_out,axis=1,output_type=tf.int32)
not_eq=tf.cast(tf.not_equal(argmax,y),tf.float32)
test_error = tf.reduce_mean(not_eq)


optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init_global_variables = tf.global_variables_initializer()


with tf.Session() as sess:
    
    sess.run(init_global_variables)
    
    if print_out_variables:
        print_out_all_variables_AND_all_trainable_variables(sess)
    
    for epoch in range(num_epochs):
        
         # permute the training data
        inds = rng.permutation(trainx_white.shape[0])
        trainx_white = trainx_white[inds]
        trainy = trainy[inds]
        
        if epoch==0:
            sess.run(init_forward,feed_dict={x_init: trainx_white[:init_batch_size]})
            # reshuffle the training data
            inds = rng.permutation(trainx_white.shape[0])
            trainx_white = trainx_white[inds]
            trainy = trainy[inds]
            
        
        train_err=0.
        for t in range(nr_batches_train):
            feed_dict = {x: trainx_white[t*batch_size:(t+1)*batch_size], y: trainy[t*batch_size:(t+1)*batch_size]}
            l,_=sess.run([cross_entropy,optimizer], feed_dict=feed_dict)
            train_err+=l
        train_err/=nr_batches_train
        print train_err
        
        test_err=0.
        for t in range(nr_batches_test):
            feed_dict = {x: testx_white[t*batch_size:(t+1)*batch_size], y: testy[t*batch_size:(t+1)*batch_size]}
            test_error_out=sess.run(test_error, feed_dict=feed_dict)
            test_err+=test_error_out
        test_err /= nr_batches_test
        print test_err
        
    


