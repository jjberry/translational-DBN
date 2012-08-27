import numpy as np
import cPickle as pickle
import pylab 
import matplotlib.cm as cm
import cv
from scipy.misc import imresize
import os, sys
import loadData
import trainCRFs
import runCRFs

class get_bin_smooth_handle(object):
    def __init__(self, h, x, kern):
        zerotol = 1e-14
        z = np.asarray(list(x))
        if (len(kern)==2) and (kern[1]==1):
            kernelDensityOnly = True
        else:
            kernelDensityOnly = False

        kern = kern[0]
        
        xx,zz = np.meshgrid(x,z)
        difs = xx-zz;
        kernvals = self.g(difs/h,kern);
        s0 = np.sum(kernvals, axis=1)
        
        self.kernvals = kernvals
        
        if kernelDensityOnly:  #Just find weighted mean, ignore slope
            self.s0_inv = 1./(s0 + zerotol)
            self.handle = self.kernelDensityEstimation
        else: 
            difskern = difs*kernvals
            s1 = np.sum(difskern, axis=1)
            temp = difs*difskern
            s2 = np.sum(temp, axis=1);
            s0_s2_minus_s1_s1_inv = 1. / ((s0*s2 - s1*s1)+zerotol)
            self.difskern = difskern
            self.s2 = s2
            self.s1 = s1
            self.s0_s2_minus_s1_s1_inv = s0_s2_minus_s1_s1_inv
            self.handle = self.localLinearRegression

    def g(self, x, kern):
        if kern == 0:
            gx = np.exp(-((x*x)/2))
        elif kern == 1:
            gx = 1./(1+(x*x))
        elif kern == 2:
            gx = np.maximum(0.75*(1-x*x),0)
        return gx
       
    def kernelDensityEstimation(self, y):
        y1 = np.zeros((y.shape[0], 1))
        y1[:,0] = y
        t0 = np.dot(self.kernvals, y1)
        yhat = t0*self.s0_inv
        return yhat
        
    def localLinearRegression(self, y):
        y1 = np.zeros((y.shape[0], 1))
        y1[:,0] = y
        t0 = np.dot(self.kernvals, y1)
        t1 = np.dot(self.difskern, y1)
        yhat = (self.s2*t0 - self.s1*t1) * self.s0_s2_minus_s1_s1_inv
        return yhat

def get_activation(x, W, b, ltype='sigmoid'):
    act = np.dot(x, W) + b
    if ltype == 'sigmoid':
        return 1. / (1. + np.exp(-act))
    else:
        return act

def run_through_network(x, network, layer_types):
    act = None
    assert (len(layer_types) == len(network))
    for i in range(len(network)):
        if i == 0:
            linput = x
        else:
            linput = act
        W = network[i].W.get_value(borrow=True)
        b = network[i].b.get_value(borrow=True)
        act = get_activation(linput, W, b, layer_types[i])
    return act
        
def contour_from_image(img):
    height = img.shape[0]
    vals = np.max(img, axis=0)
    temp = 0.05
    inds = np.zeros((1,height))
    inds[0,:] = np.arange(height)
    inds = np.dot(inds, np.exp(img/temp)) 
    inds /= np.sum(np.exp(img/temp), axis=0)
    inds1 = np.zeros((inds.shape[1],))
    inds1 = inds[0,:]
    h = 1
    kern = [0,0]
    smallcx = np.nonzero(vals>0.3)[0]
    smallcx = np.arange(smallcx.min(), smallcx.max()+1)
    f = get_bin_smooth_handle(h, smallcx, kern)
    smallcy = np.diag(f.handle(inds1[smallcx]-height) + height)

    return smallcx, smallcy

def loadImages(imgdir, h, w, m, s, sigmoid):
    if os.path.isfile(os.path.join(imgdir, 'ROI_config.txt')):
        roi_file = os.path.join(imgdir, 'ROI_config.txt')
    else:    
        if imgdir[-1] == '/':
            listclip = -2
        else:
            listclip = -1
        roi_file = os.path.join('/'.join(imgdir.split('/')[:listclip]), 
                                'ROI_config.txt')
    if os.path.isfile(roi_file):
        c = open(roi_file, 'r').readlines()
        top = int(c[1][:-1].split('\t')[1])
        bottom = int(c[2][:-1].split('\t')[1])
        left = int(c[3][:-1].split('\t')[1])
        right = int(c[4][:-1].split('\t')[1])
    else:
        top = 140
        bottom = 320
        left = 250
        right = 580   
    files = sorted(os.listdir(imgdir))
    imagenames = []
    images = []
    for i in range(len(files)):
       if files[i][-3:] == 'jpg':
           imagenames.append(files[i])
    for i in range(len(imagenames)):
        img = cv.LoadImageM(os.path.join(imgdir, imagenames[i]), iscolor=False)
        img = np.asarray(img)
        cropped = img[top:bottom, left:right]
        resized = imresize(cropped, (h, w), interp='bicubic')
        scaled = np.double(resized)/255
        raster = scaled.reshape((h*w,))
        images.append(raster)
    images = np.asarray(images)
    print images.shape
    print (m[:h*w]).shape, (s[:h*w]).shape
    if sigmoid == True:
        return images
    else:
        images = (images - m[:h*w]) / s[:h*w]
        return images            
    
def autotrace(filename, imgdir, make_contimgs=True, make_overlays=True):
    d = pickle.load(file(filename, 'rb'))
    network = d['network']
    data = d['data']
    types = d['types']
    
    if types[0] == 'sigmoid':
        sigmoid =True
    else:
        sigmoid = False

    images = loadImages(imgdir, data.height, data.width, data.m, data.s, sigmoid)

    preds = run_through_network(images, network, types) 
    contresults = []
    
    fnames = sorted(os.listdir(imgdir))
    
    for i in range(images.shape[0]):
        contresult = np.zeros((data.height*data.width,))
        contresult[data.continds] = \
                preds[i]*data.s[data.height*data.width:] +\
                data.m[data.height*data.width:]
        contresult = contresult.reshape((data.height, data.width))
        crm = np.mean(contresult)
        contresult[contresult < crm] = crm 
        contresult -= crm
        contresult /= np.max(contresult)
        contresults.append(contresult)

        if make_contimgs == True:
            print i
            picname = 'contour_%03d.jpg' % i
            pylab.imshow(contresult, cm.gray)
            pylab.savefig(picname)

            pylab.clf() #these two lines avoid the memory problem
            pylab.close()
        
        smallcx, smallcy = contour_from_image(contresult)

        interprows = data.maxy-data.miny+1
        interpcols = data.maxx-data.minx+1
        
        largecx = (smallcx-.5)*interpcols/data.width+data.minx-1
        largecy = (smallcy-.5)*interprows/data.height+data.miny-1  
        #print largecx, largecy

        if make_overlays == True:        
            img = cv.LoadImageM(os.path.join(imgdir, fnames[i]), iscolor=False)
            img = np.asarray(img, dtype=np.float)
            img /= 255
            pylab.imshow(img, cm.gray)
            pylab.plot(largecx, largecy, 'g-')
            savename = 'overlay_%03d.jpg' % i
            pylab.savefig(savename)
            pylab.clf()
            pylab.close()

    return contresults    

class CRFTrainer(object):
    def __init__(self, network, imgdir):
        d = pickle.load(file(network, 'rb'))
        self.data = d['data']
        self.contimgs = autotrace(network, os.path.join(imgdir, "JPG"))
        self.height, self.width = self.contimgs[0].shape
        self.imgdir = imgdir

    def format_crfsgd(self):
        #get sequences from imgdir filenames
        files = sorted(os.listdir(os.path.join(self.imgdir, "JPG")))
        cur_stem = files[0].split('_')[0]
        sequences = []
        sequence = []
        for i in range(len(files)):
            if files[i][-3:] == 'jpg':
                stem = files[i].split('_')[0]
                if stem == cur_stem:
                    sequence.append(files[i])
                else:
                    sequences.append(sequence)
                    sequence = [files[i]]
                    cur_stem = stem
        sequences.append(sequence)
        index = 0
        indices = []
        for i in range(len(sequences)):
            cur_index = []
            for j in range(len(sequences[i])):
                cur_index.append(index)
                index += 1
            indices.append(cur_index)
        rp = np.arange(len(sequences))
        np.random.shuffle(rp)
    
        #get the true images
        l = loadData.Loader(self.imgdir)
        l.loadContours()
        l.cleanContours()
        l.makeContourImages()
        labels = []
        for i in range(len(l.contimgs)):
            label = imresize(l.contimgs[i], (self.height, self.width), 
                    interp='bicubic')
            label = np.double(label)/255
            labels.append(label)
    
        #make the template file
        o = open('crftemplate.txt', 'w')
        for i in range(self.height*self.width):
            fnum = "%03d" %i
            rnum = str(i)
            o.write("U" + fnum + ":%x[0," + rnum + "]\n")
        o.write("\nB\n")
        o.close()
    
        def get_bin(n):
            if n < 0.2:
                b = 0
            elif n < 0.4:
                b = 1
            elif n < 0.6:
                b = 2
            elif n < 0.8:
                b = 3
            else:
                b = 4
            return b

        #make the crf training files: 1 for each column of the contimgs
        for col in range(self.width):
            print "Creating CRF file %d of %d" %(col+1, self.width)
            filename = "crf_train_col%02d.txt" %col
            output = open(filename, 'w')
            for i in range(len(sequences)):
                for j in range(len(sequences[i])):
                    img = indices[i][j]
                    t = labels[img][:,col]
                    if len(t[t>0]) == 0:
                        targ = 'none'
                    else:
                        targ = "%02d" % np.argmax(t)
                    print targ,
                    for r in range(self.height):
                        for c in range(self.width):
                            feature = get_bin(self.contimgs[img][r,c])
                            output.write("r%02dc%02df%02d\t" % (r, c, feature)) 
                    output.write("t%s\n" % targ)
                output.write("\n")
            output.close()     

    def train_crfs(self):
        trainCRFs.trainCRFs()

    def runCRFs(self):
        runCRFs.runCRFs()

    def get_crf_activations(self):
        #collect all the outputs from the results files
        results = []
        for col in range(self.width):
            filename = "results_%02d.txt" %col
            result = []
            f = open(filename, 'r').readlines()
            for line in range(len(f)):
                if f[line] != '\n':
                    result.append(f[line][:-1].split(' ')[-1])
            results.append(result)
        
        #make the images
        images = []
        for i in range(len(results[0])):
            img = np.zeros((self.height, self.width))
            for j in range(len(results)):
                act = results[j][i]
                if act != 'tnone':
                    ind = int(act[1:])
                    img[ind,j] = 1
            images.append(img)
            
        fnames = sorted(os.listdir(os.path.join(self.imgdir, "JPG")))
        for i in range(len(images)):
            print i
            smallcx, smallcy = contour_from_image(images[i])
            interprows = self.data.maxy-self.data.miny+1
            interpcols = self.data.maxx-self.data.minx+1
            largecx = (smallcx-.5)*interpcols/self.data.width+self.data.minx-1
            largecy = (smallcy-.5)*interprows/self.data.height+self.data.miny-1

            img = cv.LoadImageM(os.path.join(self.imgdir, "JPG", fnames[i]),
                    iscolor=False)
            img = np.asarray(img, dtype=np.float)
            img /= 255
            pylab.imshow(img, cm.gray)
            pylab.plot(largecx, largecy, 'g-')
            savename = 'crf_overlay_%04d.jpg' %i
            pylab.savefig(savename)
            pylab.clf()
            pylab.close()



if __name__ == "__main__":
    autotrace(sys.argv[1], sys.argv[2])
