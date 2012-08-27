import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams


class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, 
                numpy_rng=None, theano_rng=None, layer_type='sigmoid'):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.layer_type = layer_type

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W')

        if hbias is None:
            hbias = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='hbias')

        if vbias is None:
            vbias = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX), name='vbias')

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        if self.layer_type == 'sigmoid':
            return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
        elif self.layer_type == 'gaussian':
            return [pre_sigmoid_activation, pre_sigmoid_activation]
        else:
            raise ValueError('layer_type %s not implemented' % self.layer_type)
                     
    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        if self.layer_type == 'sigmoid':
            h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                                n=1, p=h1_mean,
                                                dtype=theano.config.floatX)
            return [pre_sigmoid_h1, h1_mean, h1_sample]
        elif self.layer_type == 'gaussian':
            h1_sample = self.theano_rng.normal(size=h1_mean.shape,
                                                avg=h1_mean, std=1.0,
                                                dtype=theano.config.floatX)
            return [pre_sigmoid_h1, h1_mean, h1_sample]
        else:
            raise ValueError('layer_type %s not implemented' % self.layer_type)
        
    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        if self.layer_type == 'sigmoid':
            return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
        elif self.layer_type == 'gaussian':
            return [pre_sigmoid_activation, pre_sigmoid_activation]
        else:
            raise ValueError('layer_type %s not implemented' % self.layer_type)
            
    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        if self.layer_type == 'sigmoid':
            v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                                n=1, p=v1_mean,
                                                dtype=theano.config.floatX)
            return [pre_sigmoid_v1, v1_mean, v1_sample]
        elif self.layer_type == 'gaussian':
            v1_sample = self.theano_rng.normal(size=v1_mean.shape,
                                                avg=v1_mean, std=1.0,
                                                dtype=theano.config.floatX)
            return [pre_sigmoid_v1, v1_mean, v1_sample]
        else:
            raise ValueError('layer_type %s not implemented' % self.layer_type)

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        if self.layer_type == 'sigmoid':
            cross_entropy = T.mean(
                    T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                    (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                        axis=1))
            return cross_entropy
        elif self.layer_type == 'gaussian':
            cross_entropy = T.mean(
                    T.sum(self.input * pre_sigmoid_nv +
                    (1 - self.input) * (1 - pre_sigmoid_nv),
                        axis=1))
            return cross_entropy
        else:
            raise ValueError('layer_type %s not implemented' % self.layer_type)
        
class tRBM(RBM):
    ''' Implements the translational RBM (Fasel & Berry 2010)
        previous_W is the weight matrix from the RBM pre-trained on joint sensors and labels
        previous_hbias is the hbias " " "
        input is the *joint* sensor - label input
        n_visible is the size of the visible layer of the *new* tRBM, i.e. without labels
    '''
    def __init__(self, previous_W, previous_hbias, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, 
                numpy_rng=None, theano_rng=None, layer_type='sigmoid'):
        super(tRBM, self).__init__(input, n_visible, n_hidden, W, hbias, vbias, numpy_rng, theano_rng, layer_type)
        self.previous_W = previous_W
        self.previous_hbias = previous_hbias
        self.sensors = self.input[:,0:n_visible]
        
    def t_propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.previous_W) + self.previous_hbias
        if self.layer_type == 'sigmoid':
            return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
        elif self.layer_type == 'gaussian':
            return [pre_sigmoid_activation, pre_sigmoid_activation]
        else:
            raise ValueError('layer_type %s not implemented' % layer_type)
 
    def t_sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.t_propup(v0_sample)
        if self.layer_type == 'sigmoid':
            h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                                n=1, p=h1_mean,
                                                dtype=theano.config.floatX)
            return [pre_sigmoid_h1, h1_mean, h1_sample]
        elif self.layer_type == 'gaussian':
            h1_sample = self.theano_rng.normal(size=h1_mean.shape,
                                                avg=h1_mean, std=1.0,
                                                dtype=theano.config.floatX)
            return [pre_sigmoid_h1, h1_mean, h1_sample]
        else:
            raise ValueError('layer_type %s not implemented' % layer_type)

    def get_cost_updates(self, lr=0.1, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.t_sample_h_given_v(self.input)

        chain_start = ph_sample

        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.sensors)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        if self.layer_type == 'sigmoid':
            cross_entropy = T.mean(
                    T.sum(self.sensors * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                    (1 - self.sensors) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                        axis=1))
            return cross_entropy
        elif self.layer_type == 'gaussian':
            cross_entropy = T.mean(
                    T.sum(self.sensors * pre_sigmoid_nv +
                    (1 - self.sensors) * (1 - pre_sigmoid_nv),
                        axis=1))
            return cross_entropy
        else:
            raise ValueError('layer_type %s not implemented' % layer_type)
            

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W')

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]

        
        
