import sys
import os
home = os.path.expanduser("~")
sys.path.append(os.path.join(home, 'gnumpy'))
import gnumpy as gp
import numpy as np

class RBM(object):
    ''' 
    This class implements a restricted Bolzmann machine using gnumpy,
    which runs on a gpu if cudamat is installed
    
    args:
        int n_visible:    the number of visible units
        int n_hidden:     the number of hidden units, default is n_visible
        string vistype:   type of units for visible layer, default 'sigmoid'
        string hidtype:   type of units for hidden layer, default 'sigmoid'
        array W:          the 2d weight matrix, default None
        array hbias:      the bias weights for the hidden layer, default None
        array vbias:      the bias weights for the visible layer, default None
        int batch_size:   default 128

        if W, hbias, vbias are left as None (default), they will be created and 
            initialized automatically.

    methods:
        train(int num_epochs, array hidden, bool sample)
        prop_up(array data)
        prop_down(array data)
        hidden_state(array data)

    variables:
        array wu_vh:  the weight update array which can be reused
        array wu_v:   the update array for vbias
        array wu_h:   the update array for hbias

    '''

    def __init__(self, n_visible, n_hidden=None, vistype='sigmoid', 
            hidtype='sigmoid', W=None, hbias=None, vbias=None, batch_size=128):
        # initialize parameters
        self.SIZE_LIMIT = 80000000 # the size of the largest gpu array
        self.vistype = vistype
        self.hidtype = hidtype
        self.batch_size = batch_size
        self.n_visible = n_visible
        if n_hidden is None:
            n_hidden = self.n_visible
        self.n_hidden = n_hidden
        n = self.n_visible*self.n_hidden + self.n_hidden
        bound = 2.38 / np.sqrt(n)
        if W is None:
            W = np.zeros((self.n_visible, self.n_hidden))
            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    W[i,j] = np.random.uniform(-bound, bound)
        W = gp.garray(W)
        self.W = W
        if vbias is None:
            vbias = gp.zeros(self.n_visible)
        else:
            vbias = gp.garray(vbias)
        self.vbias = vbias
        if hbias is None:
            hbias = np.zeros((self.n_hidden,))
            for i in range(self.n_hidden):
                hbias[i] = np.random.uniform(-bound, bound)
        hbias = gp.garray(hbias)
        self.hbias = hbias
        #initialize updates
        self.wu_vh = gp.zeros((self.n_visible, self.n_hidden))
        self.wu_v = gp.zeros(self.n_visible)
        self.wu_h = gp.zeros(self.n_hidden)

    def train(self, fulldata, num_epochs, eta=0.01, hidden=None, sample=False, 
            early_stop=True):
        ''' 
        Method to learn the weights of the RBM.

        args: 
            array fulldata: the training data
            int num_epochs: the number of times to run through the training data
            float eta:      the learning rate, default 0.01
            array hidden:   optional array specifying the hidden representation
                            to learn (for use in a translational-RBM)
            bool sample:    specifies whether training should use sampling, 
                            default False
            bool early_stop: whether to use early stopping, default True

        '''
        if hidden is not None:
            # check that there is a hidden rep for each data row
            assert hidden.shape[0] == data.shape[0]
            # check that we have the right number of hidden units
            assert hidden.shape[1] == self.n_hidden

        # these parameters control momentum changes
        initial_momentum = 0.5
        final_momentum = 0.9
        momentum_iter = 5

        # when dealing with large arrays, we have to break the data into
        # manageable chunks to avoid out of memory err
        if fulldata.size < self.SIZE_LIMIT:
            n_chunks = 1
            chunk_size = fulldata.shape[0]
        else:
            n_chunks = int(np.ceil(fulldata.size/float(self.SIZE_LIMIT)))
            chunk_size = fulldata.shape[0]/n_chunks
        
        num_batches = chunk_size/self.batch_size
        err_hist = [] # keep track of the errors for early stopping
        for epoch in range(num_epochs):
            if epoch <= momentum_iter:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            err = []
            print "Training epoch %d of %d," %(epoch+1, num_epochs),
            for chunk in range(n_chunks):
                num_batches = chunk_size/self.batch_size
                data = gp.garray(fulldata[chunk*chunk_size:(chunk+1)*chunk_size])
                if hidden is not None:
                    hid_chunk = gp.garray(hidden[chunk*chunk_size:(chunk+1)*chunk_size])

                for batch in range(num_batches):
                    # positive phase
                    v1 = data[batch*self.batch_size:(batch+1)*self.batch_size]
                    if hidden is None:
                        h1 = self.prop_up(v1)
                    else:
                        h1 = hid_chunk[batch*self.batch_size:(batch+1)*self.batch_size]

                    # negative phase
                    if sample:
                        hSampled = h1.rand() < h1
                        v2 = self.prop_down(hSampled)
                    else:
                        v2 = self.prop_down(h1)
                    h2 = self.prop_up(v2)
                
                    # update weights
                    self.wu_vh = self.wu_vh * momentum + gp.dot(v1.T, h1) - gp.dot(v2.T, h2)
                    self.wu_v = self.wu_v * momentum + v1.sum(0) - v2.sum(0)
                    self.wu_h = self.wu_h * momentum + h1.sum(0) - h2.sum(0)

                    self.W += self.wu_vh * (eta/self.batch_size)
                    self.vbias += self.wu_v * (eta/self.batch_size)
                    self.hbias += self.wu_h * (eta/self.batch_size)
                    # calculate reconstruction error
                    err.append((v2-v1).euclid_norm()**2/(self.n_visible*self.batch_size))
                err_hist.append(np.mean(err))
                print "mean squared error: "+ str(np.mean(err))
            
            # early stopping
            if early_stop:
                recent_err = np.mean(err_hist[epoch-50:epoch])
                early_err = np.mean(err_hist[epoch-200:epoch-150])
                if (epoch > 250) and ((recent_err * 1.2) > early_err):
                    break

    def prop_up(self, data):
        '''
        Method to return the hidden representation given data on the visible layer.

        args:
            array data:         the data on the visible layer
        returns:
            array hid:   the probabilisitic activation of the hidden layer
        
        '''
        hid = gp.dot(data, self.W) + self.hbias
        if self.hidtype == 'sigmoid':
            return hid.logistic()
        else:
            return hid

    def prop_down(self, data):
        '''
        Method to return the visible representation given the hidden

        args:
            array data:         the hidden representation
        returns:
            array vis:   the activation of the visible layer
        '''
        vis = gp.dot(data, self.W.T) + self.vbias
        if self.vistype == 'sigmoid':
            return vis.logistic()
        else:
            return vis

    def hidden_state(self, data):
        '''
        Method to sample from the hidden representation given the visible

        args:
            array data:  the data on the visible layer
        returns:
            array hSampled: the binary representation of the hidden layer activation
        '''
        hid = self.prop_up(data)
        hSampled = hid.rand() < hid
        return hSampled

class Holder(object):
    '''
    Objects of this class hold values of the RBMs in numpy arrays to free up space 
    on the GPU
    '''
    def __init__(self, rbm):
        self.W = rbm.W.as_numpy_array()
        self.hbias = rbm.hbias.as_numpy_array()
        self.vbias = rbm.vbias.as_numpy_array()
        self.n_hidden = rbm.n_hidden
        self.n_visible = rbm.n_visible
        self.hidtype = rbm.hidtype
        self.vistype = rbm.vistype

    def prop_up(self, data):
        hid = np.dot(data, self.W) + self.hbias
        if self.hidtype == 'sigmoid':
            return 1./(1. + np.exp(-hid)) 
        else:
            return hid

class DeepNet(object):
    '''
    A class to implement a deep neural network

    args:
        list[int] layer_sizes: defines the number and size of layers 
        list[str] layer_types: defines layer types, 'sigmoid' or 'gaussian'

    methods: 
        train
        run_through_network
    '''
    def __init__(self, layer_sizes, layer_types):
        assert len(layer_sizes) == len(layer_types)
        self.layer_sizes = layer_sizes
        self.layer_types = layer_types
        
    def train(self, data, epochs, eta):
        '''
        Trains the deep net one RBM at a time

        args:
            array data:         the training data (a gnumpy.array)
            list[int] epochs:   the number of training epochs for each RBM
            float eta:          the learning rate
        '''
        layers = []
        vis = data
        for i in range(len(self.layer_sizes)-1):
            print "Pretraining RBM %d, vis=%d, hid=%d" % (i+1, self.layer_sizes[i],
                    self.layer_sizes[i+1])
            g_rbm = RBM(self.layer_sizes[i], self.layer_sizes[i+1], 
                    self.layer_types[i], self.layer_types[i+1])
            g_rbm.train(vis, epochs[i], eta)
            hid = self.get_activation(g_rbm, vis)
            vis = hid
            n_rbm = Holder(g_rbm)
            layers.append(n_rbm)
            gp.free_reuse_cache()
        self.network = layers

    def get_activation(self, rbm, data):
        # trying to prop_up the whole data set causes out of memory err
        hid = np.zeros((data.shape[0], rbm.n_hidden))
        breaks = range(0, hid.shape[0], 128)
        breaks.append(hid.shape[0])
        for i in range(len(breaks)-1):
            hid[breaks[i]:breaks[i+1]] = \
                    (rbm.prop_up(data[breaks[i]:breaks[i+1]])).as_numpy_array()
        return hid

    def run_through_network(self, data):
        hid = data
        for n_rbm in self.network:
            vis = gp.garray(hid)
            g_rbm = RBM(n_rbm.n_visible, n_rbm.n_hidden, n_rbm.vistype, 
                    n_rbm.hidtype, n_rbm.W, n_rbm.hbias, n_rbm.vbias)
            hid = self.get_activation(g_rbm, data)
            gp.free_reuse_cache()
        return hid


if __name__ == "__main__":
    data = np.load('scaled_images.npy')
    data = np.asarray(data, dtype='float32')
    data /= 255.0
    #m = data.mean(0)
    #s = data.std(0)
    #data = (data - m)/s
    #data = gp.garray(data)
    t = DeepNet([data.shape[1], data.shape[1], data.shape[1], data.shape[1]*2],
            ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'])
    t.train(data, [5, 5, 5], 0.0025)
    out = t.run_through_network(data)
    print out.shape
    np.save('output.npy', out)
    
