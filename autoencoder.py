import numpy as np
import deepnet
import backprop
import cPickle as pickle
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def demo_autoencoder():
    #load and norm the data
    data = np.load('scaled_images.npy')
    data = np.asarray(data, dtype='float32')
    data /= 255.0
    #set up and train the initial deepnet
    dnn = deepnet.DeepNet([data.shape[1], data.shape[1], data.shape[1], 
        42], ['sigmoid','sigmoid','sigmoid','sigmoid'])
    dnn.train(data, [225, 75, 75], 0.0025)
    #save the trained deepnet
    pickle.dump(dnn, file('pretrained.pkl','wb'))
    #unroll the deepnet into an autoencoder
    autoenc = unroll_network(dnn.network)
    ##fine-tune with backprop
    mlp = backprop.NeuralNet(network=autoenc)
    trained = mlp.train(mlp.network, data, data, max_iter=30, 
            validErrFunc='reconstruction', targetCost='linSquaredErr')
    ##save
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
        elayer = backprop.Layer(network[i].W.T, network[i].hbias, network[i].n_hidden, network[i].hidtype)
        dlayer = backprop.Layer(network[i].W, network[i].vbias, network[i].n_visible, network[i].vistype)
        encoder.append(elayer)
        decoder.append(dlayer)
    decoder.reverse()
    encoder.extend(decoder)
    return encoder

def save_net_as_mat(pickled_net):
    '''
    Takes the network pickle file saved in demo_autoencoder and saves it as a .mat
    file for use with matlab
    '''
    network = pickle.load(file(pickled_net,'rb'))
    mdic = {}
    for i in range(len(network)/2):
        mdic['W%d'%(i+1)] = network[i].W.as_numpy_array()
        mdic['b%d'%(i+1)] = network[i].hbias.as_numpy_array()
        mdic['hidtype%d'%(i+1)] = network[i].hidtype
    scipy.io.savemat('network.mat', mdic)

def visualize_results(netfile, datafile):
    network = pickle.load(file(netfile, 'rb'))
    #network = unroll_network(dnn.network)
    data = np.load(datafile)
    data = np.asarray(data, dtype='float32')
    data /= 255.0
    mlp = backprop.NeuralNet(network=network)
    recon = mlp.run_through_network(data, network)
    inds = np.arange(recon.shape[0])
    np.random.shuffle(inds)
    for i in range(10):
        dim = int(np.sqrt(data.shape[1]))
        orig = data[inds[i]].reshape((dim,dim))
        rec = recon[inds[i]].reshape((dim,dim))
        plt.figure(i)
        ax = plt.subplot(211)
        plt.imshow(orig, cmap=cm.gray)
        ax.set_yticks([])
        ax.set_xticks([])
        ax = plt.subplot(212)
        plt.imshow(rec, cmap=cm.gray)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.savefig('img_%d.jpg'%(inds[i]))

if __name__ == "__main__":
    demo_autoencoder()
    visualize_results('network.pkl','scaled_images.npy')

