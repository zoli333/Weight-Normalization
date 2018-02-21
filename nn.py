import numpy as np
import tensorflow as tf


def int_shape(x):
    return list(map(int, x.get_shape()))
    

def mean_only_batch_norm_impl(x, pop_mean, b, is_conv_out=True, deterministic=False, decay=0.9, name='meanOnlyBatchNormalization'):
    '''
    input comes in which is t=(g*V/||V||)*x
    deterministic : separates training and testing phases
    '''
    with tf.variable_scope(name):
        if deterministic:
            # testing phase, return the result with the accumulated batch mean
            return x - pop_mean + b
        else:
            # compute the current minibatch mean
            if is_conv_out:
                # using convolutional layer as input
                m, _ = tf.nn.moments(x, [0,1,2])
            else:
                # using fully connected layer as input
                m, _ = tf.nn.moments(x, [0])
            # update minibatch mean variable
            pop_mean_op = tf.assign(pop_mean, pop_mean * decay + m * (1 - decay))
            with tf.control_dependencies([pop_mean_op]):
                return x - m + b
            
    
    


def batch_norm_impl(x,is_conv_out=True, deterministic=False, decay=0.9, name='BatchNormalization'):
    with tf.variable_scope(name):
        scale = tf.get_variable('scale',shape=x.get_shape()[-1],dtype=tf.float32,initializer=tf.ones_initializer(),trainable=True)
        beta = tf.get_variable('beta',shape=x.get_shape()[-1],dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=True)
        pop_mean = tf.get_variable('pop_mean',shape=x.get_shape()[-1],dtype=tf.float32,initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var',shape=x.get_shape()[-1],dtype=tf.float32,initializer=tf.ones_initializer(), trainable=False)
        
        if deterministic:
            return tf.nn.batch_normalization(x,pop_mean,pop_var,beta,scale,0.001)
        else:
            if is_conv_out:
                batch_mean, batch_var = tf.nn.moments(x,[0,1,2])
            else:
                batch_mean, batch_var = tf.nn.moments(x,[0])
            
            pop_mean_op = tf.assign(pop_mean,
                                    pop_mean * decay + batch_mean * (1 - decay))
            pop_var_op = tf.assign(pop_var,
                                    pop_var * decay + batch_var * (1 - decay))
            
            with tf.control_dependencies([pop_mean_op, pop_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 0.001)
            
            
            

def conv2d(x, num_filters, filter_size=[3,3],  pad='SAME', stride=[1,1], nonlinearity=None, init_scale=1., init=False, 
            use_weight_normalization=False, use_batch_normalization=False, use_mean_only_batch_normalization=False,
            deterministic=False,name=''):
                
    '''
    deterministic : used for batch normalizations (separates the training and testing phases)
    '''
    
    with tf.variable_scope(name):
        V = tf.get_variable('V', shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        
        
        if use_batch_normalization is False: # not using bias term when doing batch normalization, avoid indefinit growing of the bias, according to BN2015 paper
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.), trainable=True)
        
        if use_mean_only_batch_normalization:
            pop_mean = tf.get_variable('meanOnlyBatchNormalization/pop_mean',shape=[num_filters], dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=False)
        
        if use_weight_normalization:
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                  initializer=tf.constant_initializer(1.), trainable=True)
            
            if init:
                v_norm = tf.nn.l2_normalize(V,[0,1,2])
                x = tf.nn.conv2d(x, v_norm, strides=[1] + stride + [1],padding=pad)
                m_init, v_init = tf.nn.moments(x, [0,1,2])
                scale_init=init_scale/tf.sqrt(v_init + 1e-08)
                g = g.assign(scale_init)
                b = b.assign(-m_init*scale_init)
                x = tf.reshape(scale_init,[1,1,1,num_filters])*(x-tf.reshape(m_init,[1,1,1,num_filters]))
            else:
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
                if use_mean_only_batch_normalization: # use weight-normalization combined with mean-only-batch-normalization
                    x = tf.nn.conv2d(x,W,strides=[1]+stride+[1],padding=pad)
                    x = mean_only_batch_norm_impl(x,pop_mean,b,is_conv_out=True, deterministic=deterministic)
                else:
                    # use just weight-normalization
                    x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)
                
        elif use_batch_normalization:
            x = tf.nn.conv2d(x,V,[1]+stride+[1],pad)
            x = batch_norm_impl(x,is_conv_out=True,deterministic=deterministic)
        else:
            x = tf.nn.bias_add(tf.nn.conv2d(x,V,strides=[1]+stride+[1],padding=pad),b)
            
        
        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)
        
        return x
        
        
        



def dense(x, num_units, nonlinearity=None, init_scale=1., init=False, 
                use_weight_normalization=False, use_batch_normalization=False,
                use_mean_only_batch_normalization=False,
                deterministic=False,name=''):
    
    with tf.variable_scope(name):
        V = tf.get_variable('V', shape=[int(x.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        
        if use_batch_normalization is False: # not using bias term when doing basic batch-normalization, avoid indefinit growing of the bias, according to BN2015 paper
            b = tf.get_variable('b', shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)
        if use_mean_only_batch_normalization:
            pop_mean = tf.get_variable('pop_mean',shape=[num_units], dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=False)
            
        if use_weight_normalization:
            g = tf.get_variable('g', shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
            if init:
                v_norm = tf.nn.l2_normalize(V,[0])
                x = tf.matmul(x, v_norm)
                m_init, v_init = tf.nn.moments(x, [0])
                scale_init = init_scale/tf.sqrt(v_init + 1e-10)
                g = g.assign(scale_init)
                b = b.assign(-m_init * scale_init)
                x = tf.reshape(scale_init,[1,num_units])*(x-tf.reshape(m_init,[1,num_units]))
            else:
                x = tf.matmul(x, V)
                scaler = g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))
                x = tf.reshape(scaler,[1,num_units])*x
                b = tf.reshape(b,[1,num_units])
                if use_mean_only_batch_normalization:
                    x = mean_only_batch_norm_impl(x, pop_mean, b, is_conv_out=False, deterministic=deterministic)
                else:
                    x = x + b
                
        elif use_batch_normalization:
            x = tf.matmul(x, V)
            x = batch_norm_impl(x,is_conv_out=False,deterministic=deterministic)
        else:
            x = tf.nn.bias_add(tf.nn.conv2d(x,V,strides=[1,stride,stride,1],padding=padding),b)
        
    
        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x



def lRelu(x, alpha=0.1, name='leakyRelu'):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x,alpha=alpha)



def gaussian_noise(x, sigma=0.15, deterministic=False, name=''):
    with tf.variable_scope(name):
        if deterministic:
            return x
        else:
            noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32) 
            return x + noise


def globalAvgPool(x, name=''):
    with tf.variable_scope(name):
        return tf.reduce_mean(x, axis=[1,2]) # input dim: (batch,h,w,c), take the mean on (h,w) dimensions


def NiN(x, num_units, nonlinearity=None, init=False,use_weight_normalization=False, use_batch_normalization=False,
                use_mean_only_batch_normalization=False,deterministic=False,name=''):
    """ a network in network layer (1x1 CONV) """
    with tf.variable_scope(name):
        s = int_shape(x)
        x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
        x = dense(x, num_units=num_units, nonlinearity=nonlinearity,init=init,
                                            use_weight_normalization=use_weight_normalization,
                                            use_batch_normalization=use_batch_normalization,
                                            use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                            deterministic=deterministic,
                                            name=name)
        
        return tf.reshape(x, s[:-1]+[num_units])


def dropout(x, keep_prob=0.5, deterministic=False, name=''):
    with tf.variable_scope(name):
        if deterministic:
            return x
        else:
            x = tf.nn.dropout(x,keep_prob=keep_prob,name=name)
            return x





'''
References:

-BN2015 paper
    https://arxiv.org/pdf/1502.03167v3.pdf
    
-Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    https://arxiv.org/pdf/1602.07868.pdf

'''
