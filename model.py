import tensorflow as tf
import nn

def model_spec(x, keep_prob=0.5, deterministic=False, init=False, use_weight_normalization=False, use_batch_normalization=False, use_mean_only_batch_normalization=False):
    x = nn.gaussian_noise(x,deterministic=deterministic,name='gaussian_noise')
    
    x = nn.conv2d(x,num_filters=96,init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv1',nonlinearity=nn.lRelu)
                                             
    x = nn.conv2d(x,num_filters=96,init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv2',nonlinearity=nn.lRelu)
                                             
    x = nn.conv2d(x,num_filters=96,init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv3',nonlinearity=nn.lRelu)
                                             
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='max_pool_1')
    x = nn.dropout(x,keep_prob=keep_prob,deterministic=deterministic,name='drop1')
    
    x = nn.conv2d(x,num_filters=192,init=init,use_weight_normalization=use_weight_normalization,
                                              use_batch_normalization=use_batch_normalization,
                                              use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                              deterministic=deterministic,
                                              name='conv4',nonlinearity=nn.lRelu)
                                             
    x = nn.conv2d(x,num_filters=192,init=init,use_weight_normalization=use_weight_normalization,
                                              use_batch_normalization=use_batch_normalization,
                                              use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                              deterministic=deterministic,
                                              name='conv5',nonlinearity=nn.lRelu)
                                             
    x = nn.conv2d(x,num_filters=192,init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv6',nonlinearity=nn.lRelu)
                                             
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='max_pool_2')
    x = nn.dropout(x,keep_prob=keep_prob,deterministic=deterministic,name='drop2')
    
    
    x = nn.conv2d(x,num_filters=192,init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             pad='VALID',name='conv7',nonlinearity=nn.lRelu)
    
    
    x = nn.NiN(x,num_units=192,nonlinearity=nn.lRelu, init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,name='Nin1')
                                             
    x = nn.NiN(x,num_units=192,nonlinearity=nn.lRelu,init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,name='Nin2')
    
    x = nn.globalAvgPool(x,name='Globalavgpool1')
    
    x = nn.dense(x, num_units=10, nonlinearity=None, init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='output_dense')
    
    
    return x
    
    
    
    
    
    
    


