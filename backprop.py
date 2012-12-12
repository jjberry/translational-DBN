import sys
import os
home = os.path.expanduser("~")
sys.path.append(os.path.join(home, 'gnumpy'))
import gnumpy as gp
import numpy as np
import scipy
import scipy.optimize
import deepnet

class NeuralNet(object):
    '''
    Implementation of a Multi-Layer Perception trained by backprop. This class 
    accepts pre-trained networks for use in a deep neural network. Pre-trained 
    nets should consist of a list of objects, where each object has a W and hbias
    variables containing numpy arrays, n_hidden containing the number of 
    hidden units, and hidtype containing a string with the activation type, i.e.
    "sigmoid".
    '''
    def __init__(self, network=None, layer_sizes=None, layer_types=None):
        layers = []
        if (network != None):
            # copy the weights from the given network onto the GPU
            for rbm in network:
                l = Layer(rbm.W, rbm.hbias, rbm.n_hidden, rbm.hidtype)
                layers.append(l)
        else:
            # if no pre-trained network is given, initialize random weights
            assert layer_sizes is not None
            assert layer_types is not None
            assert len(layer_sizes) == len(layer_types)
            # randomize the network weights according to the Bottou proposition
            # this is borrowed from the ffnet project:
            # http://ffnet.sourceforge.net/_modules/ffnet.html#ffnet.randomweights
            n = 0
            for i in range(len(layer_sizes)-1):
                n += layer_sizes[i]*layer_sizes[i+1]
                n += layer_sizes[i+1]
            bound = 2.38 / np.sqrt(n)
            for i in range(len(layer_sizes)-1):
                W = np.zeros((layer_sizes[i+1]*layer_sizes[i],))
                for j in range(W.size):
                    W[j] = np.random.uniform(-bound, bound)
                W = W.reshape((layer_sizes[i+1], layer_sizes[i]))
                hbias = np.zeros((layer_sizes[i+1],))
                for j in range(hbias.size):
                    hbias[j] = np.random.uniform(-bound, bound)
                hbias = hbias.reshape((layer_sizes[i+1],1))
                l = Layer(W, hbias, layer_sizes[i+1], layer_types[i+1])
                layers.append(l)
        self.network = layers

    def run_through_network(self, data, net=None):
        '''
        Gets the output of the top layer of the network given input data on the 
        bottom.

        args:   
            array data: the input data
            obj net:    the network to use, default is self.network
        returns:
            array hid:  the activation of the top layer 
        '''
        if net is None:
            net = self.network
        hid = data
        for layer in net:
            vis = gp.garray(hid)
            hid = self.get_activation(layer, vis)
            gp.free_reuse_cache()
        return hid

    def get_activation(self, layer, data):
        '''
        Gets the activation of a single layer given input data

        args:
            obj layer:  the layer object 
            array data: the input data
        returns:
            array hid:  the output of the layer
        '''
        if not hasattr(layer, 'n_hidden'):
            layer = layer[0]
        hid = np.zeros((data.shape[0], layer.n_hidden))
        breaks = range(0, hid.shape[0], 128)
        breaks.append(hid.shape[0])
        for i in range(len(breaks)-1):
            s = breaks[i]
            e = breaks[i+1]
            act = gp.dot(data[s:e], layer.W.T) + layer.hbias.T
            if layer.hidtype == 'sigmoid':
                hid[s:e] = (act.logistic()).as_numpy_array()
            else:
                hid[s:e] = act.as_numpy_array()
        return hid

    def train(self, network, data, targets, validX=None, validT=None, max_iter=100,
            validErrFunc='classification', targetCost='linSquaredErr', initialfit=5,
            cg_iter=20):
        '''
        Trains the network using backprop

        args:
            list[obj] network: the network
            array data:     the training data
            array targets:  the training labels
            array validX:   the validation data (optional)
            array validT:   the validation labels (optional)
            int max_iter:   the maximum number of backprop iterations
            string validErrFunc: determines which kind of network to train, 
                            i.e. classification or reconstruction
            string targetCost:  determines which cost function to use, i.e.
                            linSquaredErr, crossEntropy, or softMax
                            linSquaredErr works only for gaussian output units
                            softmax works only for exp output units (not implemented)
            int initialfit: if n>0, top layer only will be trained for n iterations
            int cg_iter:    the max number of iterations for conjugate gradient
                            optimization, default=20
        '''
        # initialize parameteres
        self.validErrFunc = validErrFunc
        self.targetCost = targetCost
        self.n, self.m = data.shape
        self.cg_iter = cg_iter
        numunits = 0
        for i in range(len(self.network)):
            numunits = numunits + self.network[i].W.shape[1] + \
                    self.network[i].hbias.shape[0]
        self.numunits = numunits
        self.batch_size = 1024
        self.weights = np.ones((self.n,1))
        
        # For estimating test error
        tindex = np.arange(self.n)
        np.random.shuffle(tindex)
        tinds = tindex[:(np.min([self.batch_size, self.n]))]
        
        # Perform gradient descent
        print "Starting %d iterations of backprop." % max_iter
        if (initialfit>0):  
            # This gets the activation of next to last layer to train top layer 
            transformedX = self.run_through_network(data, network[:-1])
        for i in range(max_iter):
            trainerr = self.getError(network, data[tinds,:], targets[tinds,:],
                    self.weights[tinds])
            if validX is not None:
                validerr = self.getError(network, validX, validT, 
                        np.ones((validX.shape[0],)))
                print "Iteration %3d: TrainErr = %4.3f, ValidErr = %4.3f" % \
                        (i+1, trainerr, validerr)
            else:
                print "Iteration %3d: TrainErr = %4.3f" %(i+1, trainerr)
            # Train the top layer only for initialfit iters
            if (i < initialfit):
                toplayer = self.doBackprop(transformedX, targets, [network[-1]])
                network[-1] = toplayer[0]
            else:
                network = self.doBackprop(data, targets, network)

        # Print the final training error
        trainerr = self.getError(network, data[tinds,:], targets[tinds,:],
                self.weights[tinds])
        if validX is not None:
            validerr = self.getError(network, validX, validT, 
                    np.ones((validX.shape[0],)))
            print "Final        : TrainErr = %4.3f, ValidErr = %4.3f" % \
                    (trainerr, validerr)
        else:
            print "Final        : TrainErr = %4.3f" %(trainerr)

        return network

    def getError(self, network, X, T, weights):
        '''
        Calculates the error for either classification or reconstruction during
        backprop

        args:
            list[obj] network:    the network to use
            X:                    the input data
            T:                    the input targets
            weights:              weights used for backprop

        This function is designed to be called by the train() method
        '''
        err = 0
        result = self.run_through_network(X, network)
        if self.validErrFunc == 'classification':
            for i in range(X.shape[0]):
                ind = np.argmax(result[i,:])
                targ = np.argmax(T[i,:])
                if ind != targ:
                    err = err + weights[i]
        else:
            for i in range(X.shape[0]):
                err = err + np.sqrt(np.sum(np.square(result[i,:]-T[i,:])))*weights[i]
        validerr = err / np.sum(weights)
        return validerr

    def doBackprop(self, data, targets, network):
        '''
        Executes 1 iteration of backprop

        args:
            array data:         the training data
            array targets:      the training targets
            list[obj] network:  the network
        This function is designed to be called by the train() method
        '''
        no_layers = len(network)
        index = np.arange(self.n)
        np.random.shuffle(index)
        nbatches = len(range(0,self.n, self.batch_size))
        count = 0 
        for batch in range(0, self.n, self.batch_size):
            if batch + 2*self.batch_size > self.n:
                batchend = self.n
            else:
                batchend = batch + self.batch_size
            # Select current batch
            tmpX = data[index[batch:batchend],:]
            tmpT = targets[index[batch:batchend],:]
            tmpW = self.weights[index[batch:batchend],:]

            # flatten out the weights and store them in v
            v = []
            for i in range(no_layers):
                w = network[i].W.as_numpy_array()
                b = network[i].hbias.as_numpy_array()
                v.extend((w.reshape((w.shape[0]*w.shape[1],))).tolist())
                v.extend((b.reshape((b.shape[0]*b.shape[1],))).tolist())
            v = np.asarray(v)

            # Conjugate gradient minimiziation
            result = scipy.optimize.minimize(self.backprop_gradient, v, 
                    args=(network, tmpX, tmpT, tmpW),
                    method='CG', jac=True, options={'maxiter': self.cg_iter})
            if (count%10 == 0):
                print "batch %d of %d. success: %s" %(count+1, nbatches, 
                     str(result.success))
            count += 1         
            v = result.x

            # unflatten v and put new weights back
            ind =0 
            for i in range(no_layers):
                h,w = network[i].W.shape
                network[i].W = gp.garray((v[ind:(ind+h*w)]).reshape((h,w)))
                ind += h*w
                b = len(network[i].hbias)
                network[i].hbias = gp.garray((v[ind:(ind+b)]).reshape((b,1)))
                ind += b

        # debugging help
        #print "=================="
        #print "W 1", network[0].W.shape
        #print network[0].W
        #print "bias 1", network[0].hbias.shape
        #print network[0].hbias
        #print "W 2", network[1].W.shape
        #print network[1].W
        #print "bias 2", network[1].hbias.shape
        #print network[1].hbias
        #print "=================="
        
        return network
    
    def backprop_gradient(self, v, network, X, targets, weights):
        '''
        Calculates the value of the cost function and the gradient for CG 
        optimization.

        args:
            array v:            the 1d vector of weights
            list[obj] network:  the network
            array X:            training data
            array targets:      the training targets
            array weights:      the backprop weights
        returns:
            array cost:         the value of the cost function
            array grad:         the value of the gradient

        This function is called by scipy's minimize function during optimization
        '''
        if len(v.shape) == 1:
            v = v.reshape((v.shape[0],1))
        # initialize variables
        n = X.shape[0]
        numHiddenLayers = len(network)

        # put the v weights back into the network
        ind =0 
        for i in range(numHiddenLayers):
            h,w = network[i].W.shape
            network[i].W = gp.garray((v[ind:(ind+h*w)]).reshape((h,w)))
            ind += h*w
            b = network[i].hbias.shape[0]
            network[i].hbias = gp.garray(v[ind:(ind+b)]).reshape((b,1))
            ind += b

        # Run data through the network, keeping activations of each layer
        acts = [X] # a list of numpy arrays
        hid = X
        for layer in network:
            vis = gp.garray(hid)
            hid = self.get_activation(layer, vis) 
            acts.append(hid)
            gp.free_reuse_cache()

        # store the gradients
        dW = []
        db = []

        # Compute the value of the cost function
        if self.targetCost == 'crossEntropy':
            # see www.stanford.edu/group/pdplab/pdphandbook/handbookch6.html
            cost = (-1.0/n) * np.sum(np.sum(targets * np.log(acts[-1]) + \
                    (1.0 - targets) * np.log(1.0 - acts[-1]), axis=1) * weights.T)
            Ix = (acts[-1] - targets) / n
        else: #self.targetCost == 'linSquaredErr':
            cost = 0.5 * np.sum(np.sum(np.square(acts[-1] - targets), axis=1) * \
                    weights.T)
            Ix = (acts[-1] - targets)
        Ix *= np.tile(weights, (1, Ix.shape[1])).reshape((Ix.shape[0],Ix.shape[1]))
        Ix = gp.garray(Ix)

        # Compute the gradients
        for i in range(numHiddenLayers-1,-1,-1):
            # augment activations with ones
            acts[i] = gp.garray(acts[i])
            acts[i] = gp.concatenate((acts[i], gp.ones((n,1))), axis=1)

            # compute delta in next layer
            delta = gp.dot(acts[i].T, Ix)

            # split delta into weights and bias parts
            dW.append(delta[:-1,:].T)
            db.append(delta[-1,:].T)

            # backpropagate the error
            if i > 0:
                if network[i-1].hidtype == 'sigmoid':
                    Ix = gp.dot(Ix,gp.concatenate((network[i].W,network[i].hbias),
                        axis=1)) * acts[i] * (1.0 - acts[i])
                elif network[i-1].hidtype == 'gaussian':
                    Ix = gp.dot(Ix,gp.concatentate((network[i].W,network[i].hbias),
                        axis=1))
                Ix = Ix[:,:-1]
            gp.free_reuse_cache()
        dW.reverse()
        db.reverse()

        # Convert gradient information
        grad = np.zeros_like(v)
        ind = 0
        for i in range(numHiddenLayers):
            grad[ind:(ind+dW[i].size)] = \
                 (dW[i].reshape((dW[i].shape[0]*dW[i].shape[1],1))).as_numpy_array()
            ind += dW[i].size
            grad[ind:(ind+db[i].size),0] = db[i].as_numpy_array()
            ind += db[i].size
        grad = grad.reshape((grad.shape[0],))
        return cost, grad  

class Layer(object):
    '''
    A hidden layer object

    args:
        array W:    the weight array
        array hbias: the bias weights
        int n_hidden: the number of hidden units
        string hidtype: the activation function "sigmoid" or "gaussian"
    '''
    def __init__(self, W, hbias, n_hidden, hidtype):
        self.W = gp.garray(W)
        # convert 1d arrays to 2d
        if len(hbias.shape) == 1:
            hbias = hbias.reshape((hbias.shape[0],1))
        self.hbias = gp.garray(hbias)
        self.n_hidden = n_hidden
        self.hidtype = hidtype
   
def demo_xor():
    '''Demonstration of backprop with classic XOR example
    '''
    data = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    targets = np.array([[0.],[1.],[1.],[0.]])
    nn = NeuralNet(layer_sizes=[2,2,1], layer_types=['sigmoid','sigmoid','sigmoid'])
    print "initial parameters"
    print "=================="
    print "W 1", nn.network[0].W.shape
    print nn.network[0].W
    print "bias 1", nn.network[0].hbias.shape
    print nn.network[0].hbias
    print "W 2", nn.network[1].W.shape
    print nn.network[1].W
    print "bias 2", nn.network[1].hbias.shape
    print nn.network[1].hbias
    print "=================="
    net = nn.train(nn.network, data, targets, max_iter=10, targetCost='crossEntropy', 
            initialfit=0, cg_iter=100)
    print "network test:"
    output = nn.run_through_network(data, net)
    print output


if __name__ == "__main__":
    demo_xor()

            
