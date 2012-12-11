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
    dnn.train(data, [5, 5, 5], 0.0025)
    #save the trained deepnet
    pickle.dump(dnn, file('pretrained.pkl','wb'))
    #unroll the deepnet into an autoencoder
    autoenc = unroll_network(dnn.network)
    #print out the sizes of the autoenc layers
    for i in range(len(autoenc)):
        print autoenc[i].W.shape
        print autoenc[i].hbias.shape
    #fine-tune with backprop
    mlp = backprop.NeuralNet(network=autoenc)
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
    for i in range(len(network)):
        elayer = backprop.Layer(network[i].W, network[i].hbias, network[i].n_hidden, network[i].hidtype)
        dlayer = backprop.Layer(network[i].W.T, network[i].vbias, network[i].n_visible, network[i].vistype)
        encoder.append(elayer)
        decoder.append(dlayer)
    decoder.reverse()
    encoder.extend(decoder)
    return encoder

if __name__ == "__main__":
    demo_autoencoder()

