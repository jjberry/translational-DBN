import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tDBN import RBM, HiddenLayer, tRBM
import loadData
import sys
import cPickle as pickle
import time

def trainNetwork(data_dir, hidden_layers_sizes=None, hidden_layers_types=None,
                nImages=None, nFolds=5, batch_size=10, k=1, 
                pretraining_epochs=100, pretrain_lr=0.0025,
                finetune_epochs=30, finetune_lr=0.0025,
                keep_genimgs=True, sigmoid_1st_layer=False):
    ''' First layer type is assumed to be gaussian, size is determined by size of
        data.
        
        data_dir: string with location of data
        hidden_layers_sizes [optional]: list of hidden layer sizes, e.g. [500, 500]
        hidden_layers_types [optional]: list of hidden layer types, e.g. ['sigmoid','sigmoid']
        
        optional params:
        nImages = None       #max_images
        nFolds = 5          
        batch_size = 10     
        k = 1               #number of Gibbs steps during training
        pretraining_epochs = 100
        pretrain_lr = 0.01
        finetune_epochs = 30
        finetune_lr = 0.01
        keep_genimgs = False  #This determines whether to finetune with the generated images on the output layer
    '''    
    data = loadData.Loader(data_dir, max_images=nImages)
    data.loadData(sigmoid_1st_layer)
    
    nData = data.XC.shape[0]
    inds = np.arange(nData)
    np.random.shuffle(inds)    

    if hidden_layers_sizes == None:
        hidden_layers_sizes = [data.XC.shape[1], data.XC.shape[1], data.XC.shape[1]*3]
    if hidden_layers_types == None:
        hidden_layers_types = ['sigmoid', 'sigmoid', 'sigmoid']

    assert len(hidden_layers_sizes) == len(hidden_layers_types)

    n_layers = len(hidden_layers_sizes)
    
    assert n_layers > 0
    
    numpy_rng = np.random.RandomState(1234)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    fold = 0
    
    # get test set
    test_dir = 'aln0000JPG/diverse-test/Subject1/'
    testd = loadData.Loader(test_dir, continds=data.continds, m=data.m, s=data.s)
    testd.loadData(sigmoid_1st_layer)        
    
    #for traininds, validinds in data.k_fold_cross_validation(inds, nFolds):
    fold += 1
    print "Fold %d ..." % fold
    #trainX = data.XC[traininds,:]
    #validX = data.XC[validinds,:]
    trainX = data.XC
    validX = testd.XC
    
    shared_x = theano.shared(np.asarray(trainX, dtype=theano.config.floatX))
    shared_sensors = theano.shared(np.asarray(trainX[:,0:(data.height*data.width)], dtype=theano.config.floatX))
    valid_sensors = theano.shared(np.asarray(validX[:,0:(data.height*data.width)], dtype=theano.config.floatX))

    if keep_genimgs==True:
        shared_labels = theano.shared(np.asarray(trainX, dtype=theano.config.floatX))
        valid_labels = theano.shared(np.asarray(validX, dtype=theano.config.floatX))
    else:
        shared_labels = theano.shared(np.asarray(trainX[:,(data.height*data.width):], dtype=theano.config.floatX)) 
        valid_labels = theano.shared(np.asarray(validX[:,(data.height*data.width):], dtype=theano.config.floatX))
    
    x = T.matrix('x')  
    y = T.matrix('y') 

    mlp_layers = []
    rbm_layers = []
    params = []
    layer_types = []
    
    print "building model..."
    for i in xrange(n_layers):
        ##-----------------------------------------##
        # build model
        ##-----------------------------------------##
        if i == 0:
            input_size = shared_x.get_value(borrow=True).shape[1]
            layer_input = x
            if sigmoid_1st_layer == True:
                layer_type = 'sigmoid'
                activation = T.nnet.sigmoid
            else:
                layer_type = 'gaussian'
                activation = None
        else:
            input_size = hidden_layers_sizes[i - 1]
            layer_input = mlp_layers[-1].output
            layer_type = 'sigmoid'
            activation = T.nnet.sigmoid
        
        layer_types.append(layer_type)
            
        mlp_layer = HiddenLayer(rng=numpy_rng,
                            input=layer_input,
                            n_in=input_size,
                            n_out=hidden_layers_sizes[i],
                            activation=activation)

        mlp_layers.append(mlp_layer)

        params.extend(mlp_layer.params)

        rbm_layer = RBM(numpy_rng=numpy_rng,
                        theano_rng=theano_rng,
                        input=layer_input,
                        n_visible=input_size,
                        n_hidden=hidden_layers_sizes[i],
                        W=mlp_layer.W,
                        hbias=mlp_layer.b,
                        layer_type=layer_type)
        rbm_layers.append(rbm_layer)

    ##-----------------------------------------##
    # pre-train
    ##-----------------------------------------##
    index = T.lscalar('index')  # index to a minibatch
    learning_rate = T.scalar('lr')  # learning rate to use

    n_batches = shared_x.get_value(borrow=True).shape[0] / batch_size
    batch_begin = index * batch_size
    batch_end = batch_begin + batch_size

    pretrain_fns = []
    for rbm in rbm_layers:
        cost, updates = rbm.get_cost_updates(learning_rate, k=k)
        fn = theano.function(inputs=[index,
                        theano.Param(learning_rate, default=0.1)],
                                outputs=cost,
                                updates=updates,
                                givens={x:
                                shared_x[batch_begin:batch_end]})
        pretrain_fns.append(fn)

    #pretrain the DBN on joint images & contours
    for i in xrange(n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_batches):
                c.append(pretrain_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)
        print rbm_layers[i].W.get_value(borrow=True).shape
    
    ##-----------------------------------------##
    # pre-train the tRBM
    ##-----------------------------------------##
    #substitute the trbm into the bottom layer of the autoencoder
    aut_mlp_layers = []
    aut_params = []

    if sigmoid_1st_layer == True:
        activation = T.nnet.sigmoid
    else:
        activation = None
    trbm_mlp = HiddenLayer(rng=numpy_rng,
                    input=x,
                    n_in=(data.height*data.width),
                    n_out=hidden_layers_sizes[0],
                    activation=activation)

    aut_mlp_layers.append(trbm_mlp)

    aut_params.extend(trbm_mlp.params)

    if sigmoid_1st_layer == True:
        layert = 'sigmoid'
    else:
        layert = 'gaussian'

    trbm = tRBM(rbm_layers[0].W, rbm_layers[0].hbias,
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=x,
                n_visible=(data.height*data.width),
                n_hidden=hidden_layers_sizes[0],
                W=trbm_mlp.W,
                hbias=trbm_mlp.b,
                layer_type=layert)
                
    cost, updates = trbm.get_cost_updates(learning_rate, k=k)
    fn = theano.function(inputs=[index,
                    theano.Param(learning_rate, default=0.1)],
                            outputs=cost,
                            updates=updates,
                            givens={x:
                            shared_x[batch_begin:batch_end]})
    for epoch in xrange(pretraining_epochs):
        c = []
        for batch_index in xrange(n_batches):
            c.append(fn(index=batch_index, lr=pretrain_lr))
        print 'Pre-training tRBM, epoch %d, cost ' % (epoch),
        print np.mean(c)
    print trbm.W.get_value(borrow=True).shape
    
    ##-----------------------------------------##    
    # build the autoencoder
    ##-----------------------------------------##
    print "building autoencoder..."
    for i in range(1,n_layers):
        mlp_layer = HiddenLayer(rng=numpy_rng,
                                input=aut_mlp_layers[i-1].output,
                                n_in=hidden_layers_sizes[i-1],
                                n_out=hidden_layers_sizes[i],
                                W=mlp_layers[i].W,
                                b=mlp_layers[i].b,
                                activation=T.nnet.sigmoid)
        aut_mlp_layers.append(mlp_layer)
        aut_params.extend(mlp_layer.params)
    
    for i in range(n_layers-1,0,-1):
        vals = mlp_layers[i].W.get_value(borrow=True).T
        W = theano.shared(value=vals, name='W') 
        mlp_layer = HiddenLayer(rng=numpy_rng,
                            input=aut_mlp_layers[-1].output,
                            n_in=hidden_layers_sizes[i],
                            n_out=hidden_layers_sizes[i-1],
                            W=W,
                            b=rbm_layers[i].vbias,
                            activation=T.nnet.sigmoid)
        aut_mlp_layers.append(mlp_layer)
        aut_params.extend(mlp_layer.params)
        layer_types.append(layer_types[i])


    if keep_genimgs==True:
        vals = mlp_layers[0].W.get_value(borrow=True).T
        bvals = rbm_layers[0].vbias.get_value(borrow=True)
    else:    
        #Add the top layer to only use the labels
        vals = (mlp_layers[0].W.get_value(borrow=True).T)[:,(data.height*data.width):]
        bvals = rbm_layers[0].vbias.get_value(borrow=True)[(data.height*data.width):]

    if sigmoid_1st_layer == True:
        activation = T.nnet.sigmoid
    else:
        activation = None
    Wt = theano.shared(value=vals, name='W') 
    bt = theano.shared(value=bvals, name='b')
    top_layer = HiddenLayer(rng=numpy_rng,
                        input=aut_mlp_layers[-1].output,
                        n_in=vals.shape[0],
                        n_out=vals.shape[1],
                        W=Wt,
                        b=bt,
                        activation=activation)
    aut_mlp_layers.append(top_layer)
    aut_params.extend(top_layer.params)
    if sigmoid_1st_layer == True:
        layer_types.append('sigmoid')
    else:
        layer_types.append('gaussian')

    print "Network size..."
    for j in range(len(aut_mlp_layers)):
        print j+1, aut_mlp_layers[j].W.get_value(borrow=True).shape, layer_types[j]

    ##-----------------------------------------##    
    # fine tune
    ##-----------------------------------------##
    print "fine-tuning..."
    p_y_given_x = T.dot(top_layer.input, top_layer.W) + top_layer.b
    #use mean square error as cost function
    finetune_cost = T.mean(T.sum(T.sqr(p_y_given_x - y), axis=1),axis=0)
    
    gparams = []
    for i in range(len(aut_params)):
        gparam = T.grad(finetune_cost, aut_params[i], disconnected_inputs='warn')
        gparams.append(gparam)
    
    updates = {}
    for param, gparam in zip(aut_params, gparams):
        updates[param] = param - gparam * learning_rate

    train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.1)],
            outputs=finetune_cost,
            updates=updates,
            givens={x: shared_sensors[index * batch_size: (index + 1) * batch_size],
                    y: shared_labels[index * batch_size: (index + 1) * batch_size]})

    errors = T.sqr(p_y_given_x - y)
    valid_score_i = theano.function([index], errors,
            givens={x: valid_sensors[index * batch_size: (index + 1) * batch_size],
                    y: valid_labels[index * batch_size: (index + 1) * batch_size]})

    # Create a function that scans the entire validation set
    n_valid_batches = valid_sensors.get_value(borrow=True).shape[0]
    n_valid_batches /= batch_size

    def valid_score():
        return [valid_score_i(i) for i in xrange(n_valid_batches)]

    n_train_batches = shared_x.get_value(borrow=True).shape[0] / batch_size    
    # early-stopping parameters
    patience = finetune_epochs * n_train_batches  # look as this many examples regardless
    patience_increase = 4.    # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                # go through this many
                                # minibatche before checking the network
                                # on the validation set; in this case we
                                # check every epoch

    best_params = None
    best_validation_loss = np.inf
    test_score = 0.

    done_looping = False
    epoch = 0

    while (epoch < finetune_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index, finetune_lr)
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = valid_score()
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break
    
    if keep_genimgs==True:
        #pull out the last layer and cut off the sensors 
        vals = aut_mlp_layers[-1].W.get_value(borrow=True)[:,(data.height*data.width):]
        bvals = aut_mlp_layers[-1].b.get_value(borrow=True)[(data.height*data.width):]
        Wt = theano.shared(value=vals, name='W') 
        bt = theano.shared(value=bvals, name='b')
        if sigmoid_1st_layer == True:
            activation = T.nnet.sigmoid
        else:
            activation = None
        top_layer = HiddenLayer(rng=numpy_rng,
                            input=aut_mlp_layers[-2].output,
                            n_in=vals.shape[0],
                            n_out=vals.shape[1],
                            W=Wt,
                            b=bt,
                            activation=activation)
        aut_mlp_layers[-1] = top_layer
        print "last layer size: ", top_layer.W.get_value(borrow=True).shape

        
    ##-----------------------------------------##
    # Save the files               
    ##-----------------------------------------##
    t = time.strftime("%Y%b%d_%H%M%S", time.gmtime()) 
    savename = "Network_%s.pkl" % t
    o = open(savename, 'wb')
    dic = { 'network': aut_mlp_layers,
            'data': data,
            'test': testd,
            'types': layer_types,
            'settings': [data_dir, hidden_layers_sizes, hidden_layers_types, 
                        nImages, nFolds, batch_size, k, pretraining_epochs, 
                        pretrain_lr, finetune_epochs, finetune_lr]}
    pickle.dump(dic, o, -1)
    o.close()

if __name__ == "__main__":
    trainNetwork(sys.argv[1], sigmoid_1st_layer=True)

