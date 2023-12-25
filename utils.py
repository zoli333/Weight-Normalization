from scipy import linalg
import numpy as np
import cPickle
import os.path



class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)
    
    
    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        self.ZCA_mat = np.dot(tmp, U.T)
        self.mean = m
    
    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return np.dot(x.reshape((s[0],np.prod(s[1:]))) - self.mean, self.ZCA_mat).reshape(s)
        
        
        

# load CIFAR-10 data
def unpickle(file):
    fo = open(file, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((10000,3,32,32)))/128.), 'y': np.array(d['labels']).astype(np.uint8)}




def load_data():
    
    whitened_dir='data/'
    trainx_white_npy_file=whitened_dir+'trainx_white.npy'
    testx_white_npy_file=whitened_dir+'testx_white.npy'
    trainx_white_label_npy_file=whitened_dir+'trainx_white_label.npy'
    testx_white_label_npy_file=whitened_dir+'testx_white_label.npy'
    
    if os.path.exists(trainx_white_npy_file) and os.path.exists(testx_white_npy_file) and os.path.exists(trainx_white_label_npy_file) and os.path.exists(testx_white_label_npy_file):
        trainx_white=np.load(trainx_white_npy_file)
        testx_white=np.load(testx_white_npy_file)
        trainy=np.load(trainx_white_label_npy_file)
        testy=np.load(testx_white_label_npy_file)
        print "Data loaded!"
        return trainx_white, testx_white, trainy, testy
    else:
        data_dir='data/cifar-10-python/cifar-10-batches-py'
        train_data = [unpickle(data_dir+'/data_batch_' + str(i)) for i in range(1,6)]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data])
        test_data = unpickle(data_dir+'/test_batch')
        testx = test_data['x']
        testy = test_data['y']
        print "applying zca-whitening..."
        # whitening
        whitener = ZCA(x=trainx)
        trainx_white = whitener.apply(trainx)
        testx_white = whitener.apply(testx)
        np.save(trainx_white_npy_file,trainx_white)
        np.save(testx_white_npy_file,testx_white)
        np.save(trainx_white_label_npy_file,trainy)
        np.save(testx_white_label_npy_file,testy)
        print "saved"
        return trainx_white, testx_white, trainy, testy

        







