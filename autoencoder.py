import numpy as np
import deepnet
import backprop
import cPickle as pickle


def demo_autoencoder():
    #load and norm the data
    data = np.load('scaled_images.npy')
    data = np.asarray(data, dtype='float32')
    data /= 255.0
    #set up and train the initial deepnet
    dnn = deepnet.DeepNet([data.shape[1], data.shape[1], data.shape[1], 
        data.shape[1]*2], ['gaussian','sigmoid','sigmoid','sigmoid'])
    dnn.train(data, [75, 75, 75, 0.0025])
    #unroll the deepnet into an autoencoder
    autoenc = unroll_network(dnn.network)
    #fine-tune with backprop
    mlp = backprop.NeuralNet(autoenc)
    trained = mlp.train(mlp.network, data, data, max_iter=100, 
            validErrFunc='reconstruction', targetCost='linSquaredErr')
    #save
    pickle.dump(trained, file('network.pkl','wb'))

def unroll_network(network):
    '''
    Takes a pre-trained network and treats it as an encoder network. The decoder
    network is constructed by inverting the encoder. The decoder is then appended
    to the input network to produce an autoencoder.
    '''
    decoder = []
    encoder = []
    for rbm in network:
        elayer = backprop.Layer(rbm.W, rbm.hbias, rbm.n_hidden, rbm.hidtype)
        dlayer = backprop.Layer(rbm.W.T, rbm.vbias, rbm.n_visible, rbm.vistype)
        encoder.append(elayer)
        decoder.append(dlayer)
    decoder.reverse()
    return encoder.extend(decoder)

if __name__ == "__main__":
    demo_autoencoder()

