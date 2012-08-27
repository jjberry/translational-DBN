import numpy as np
import os
import cv
from scipy.interpolate import interp1d
from scipy.misc import imresize
import multiprocessing
import sys

class WorkThread(multiprocessing.Process):
    def __init__(self, WorkQueue, ResultsQueue):
        super(WorkThread, self).__init__()
        self.WorkQueue = WorkQueue
        self.ResultsQueue = ResultsQueue

    def run(self):
        flag = 'ok'
        while (flag != 'stop'):
            args = self.WorkQueue.get()
            if args == None:
                flag = 'stop'
            else:
                yi = args[0]
                interprows = args[1]
                contimg = args[2]
                miny = args[3]
                h = args[4]
                ind = args[5]
                
                for j in xrange(len(yi)):
                    if not np.isnan(yi[j]):
                        for k in xrange(interprows):
                            contimg[k,j] = np.exp( -(((miny+k-1) - yi[j])/h)**2)
                
                self.ResultsQueue.put((contimg, ind))


class Loader:
    def __init__(self, data_dir, roi=None, max_images=None, num_threads=2, continds=None, m=None, s=None):
        self.jpg_dir = os.path.join(data_dir, 'JPG')
        self.contoursCSV = os.path.join(data_dir, 'TongueContours.csv')
        self.data_dir = data_dir
        self.roi = roi
        self.max_images = max_images
        self.num_threads = num_threads
        self.continds = continds
        self.m = m
        self.s = s	
                
    def loadContours(self):
        ''' Returns lists with jpg filenames, xcoords, and ycoords from the 
            TongueContours.csv file. Similar behaviour to loadContours.m
            
            computes:
                self.contfiles: filenames
                self.contx: raw x coords
                self.conty: raw y coords
        '''
        f = open(self.contoursCSV, 'r').readlines()
        self.contfiles = []
        self.contx = []
        self.conty = []
        for i in range(1,len(f)):
            cells = f[i][:-1].split('\t')
            self.contfiles.append(os.path.join(self.jpg_dir, cells[0]))
            currentx = []
            currenty = []
            for j in range(1, 65, 2):
                currentx.append(int(cells[j]))
                currenty.append(int(cells[j+1]))
            self.contx.append(currentx)
            self.conty.append(currenty)
        self.contx = np.asarray(self.contx)
        self.conty = np.asarray(self.conty)
        
    def sampleContours(self):
        ''' Similar to sampleContours.m
        '''
        inds = np.arange(len(self.contfiles))
        np.random.shuffle(inds)
        inds = inds[:self.max_images]
        self.contfiles = np.asarray(self.contfiles)[inds]
        self.contx = np.asarray(self.contx)[inds]
        self.conty = np.asarray(self.conty)[inds]
        
    def cleanContours(self):
        ''' Similar behavior to cleanContours.m - strips empty cells from
            self.contx and self.conty
            
            computes:
                self.cxc: x coords with no empty cells
                self.cyc: y coords with no empty cells
                self.minx: the global x min 
                self.miny: the global y min 
                self.maxx: the global x max 
                self.maxy: the global y max 
        '''
        self.cxc = []
        self.cyc = []
        for i in range(len(self.contx)):
            self.cxc.append(self.contx[i, self.contx[i,:]>0])
            self.cyc.append(self.conty[i, self.conty[i,:]>0])
        self.minx = np.min(self.contx[self.contx>0])
        self.miny = np.min(self.conty[self.conty>0])
        self.maxx = np.max(self.contx)
        self.maxy = np.max(self.conty)
        
    def makeContourImages(self):
        ''' Similar to makeContourImages.m - takes the cxc and cyc and makes 
            images
            
            computes:
                self.contimgs: the list of 2D contour images
        '''
        self.contimgs = []
        interprows = self.maxy-self.miny+1
        interpcols = self.maxx-self.minx+1
        h = float(interprows)/100
        
        def interp(x, y, interp_type):
            f = interp1d(x, y, kind=interp_type, bounds_error=False, fill_value=np.nan)
            yi = f(np.arange(self.minx, self.maxx))
            return yi
        WorkQueue_ = multiprocessing.Queue()
        ResultsQueue_ = multiprocessing.Queue()

        for i in range(self.num_threads):
            thread = WorkThread(WorkQueue_, ResultsQueue_)
            thread.start()
        
        nContours = len(self.cxc)
        for i in range(nContours):
            contimg = np.zeros((interprows, interpcols))
            cx = self.cxc[i]
            cy = self.cyc[i]
            yi = interp(np.double(cx), np.double(cy), 'linear')
            yi2 = interp(np.double(cx), np.double(cy), 3)
            xd = np.max(cx) - np.min(cx)
            if xd > 6:
                ind = np.arange(np.floor(.1*xd), np.ceil(.9*xd)).astype(np.int)
                yi[ind] = yi2[ind]
                
            WorkQueue_.put([yi, interprows, contimg, self.miny, h, i])
        
        for i in range(self.num_threads):
            WorkQueue_.put(None)
        
        results = []
        for i in range(nContours):
            if ((i % 100) == 0) and (i > 0):
                print "...finished %d of %d contours" % (i, nContours)

            result = ResultsQueue_.get()
            results.append(result)
        
        sortedresults = sorted(results, key = lambda r: r[1])
        self.contimgs = [i for (i,j) in sortedresults]
        
    def combineUltrasoundAndContourImages(self, sigmoid=False):
        ''' Similar to combineUltrasoundAndContourImages.m - returns an array with
            concatenated ultrasound images and their traces from makeContourImages.
            
            computes:
                self.XC: the 2D data set of rasterized ultrasound and contour images
                self.m: the mean of XC
                self.s: the sd of XC
                self.height: the height of the ultrasound image roi
                self.width: the width of the ultrasound image roi
                self.continds: the non-zero elements of contimgs
        '''
        # figure out what ROI to use
        if self.roi == None:
            if os.path.isfile(os.path.join(self.data_dir, 'ROI_config.txt')):
                print "Found ROI_config.txt"
                c = open(os.path.join(self.data_dir, 'ROI_config.txt'), 'r').readlines()
                top = int(c[1][:-1].split('\t')[1])
                bottom = int(c[2][:-1].split('\t')[1])
                left = int(c[3][:-1].split('\t')[1])
                right = int(c[4][:-1].split('\t')[1])
                print "using ROI: [%d:%d, %d:%d]" % (top, bottom, left, right)
            else:
                print "ROI_config.txt not found"
                top = 140 #default settings for the Sonosite Titan
                bottom = 320
                left = 250
                right = 580
                print "using ROI: [%d:%d, %d:%d]" % (top, bottom, left, right)
        else:
            top = self.roi[0]
            bottom = self.roi[1]
            left = self.roi[2]
            right = self.roi[3]
            print "using ROI: [%d:%d, %d:%d]" % (top, bottom, left, right)
            
        scale = 0.1
        #get height and width
        img = cv.LoadImageM(self.contfiles[0], iscolor=False)
        img = np.asarray(img)
        cropped = img[top:bottom, left:right]
        cheight, cwidth = cropped.shape
        self.height = np.floor(cheight * scale).astype(np.int)
        self.width = np.floor(cwidth * scale).astype(np.int)

        if self.continds == None:
            continds = np.arange(self.height*self.width)
            mask = np.zeros((self.height, self.width)).astype(np.bool)
            for i in range(len(self.contimgs)):
                cont = imresize(self.contimgs[i], (self.height, self.width), interp='bicubic')
                cont = np.double(cont)/255
                mask = np.logical_or(mask, cont>0.01)
            mask = mask.reshape((self.height*self.width,))
            self.continds = continds[mask]
        
        XC = np.zeros((len(self.contfiles), self.height*self.width+len(self.continds)))
        for i in range(len(self.contfiles)):
            img = cv.LoadImageM(self.contfiles[i], iscolor=False)
            img = np.asarray(img)
            cropped = img[top:bottom, left:right]
            resized = imresize(cropped, (self.height, self.width), interp='bicubic') 
            scaled = np.double(resized)/255
            
            cont = imresize(self.contimgs[i], (self.height, self.width), interp='bicubic')
            cont = np.double(cont)/255
            s = np.max(cont, axis=0)
            s[s<0.01] = 1.
            cont = cont / s
            cont[cont<0] = 0.
            
            ultrasound = scaled.reshape((self.height*self.width,))
            contour = cont.reshape((self.height*self.width,))[self.continds]
            
            XC[i,:] = np.concatenate([ultrasound, contour])
        
        if self.m == None:    
            self.m = np.mean(XC, axis=0)
            self.s = np.std(XC, axis=0)
            self.s[self.s<0.001] = 1.
        if sigmoid == False:
            self.XC = (XC-self.m)/self.s  
        else:
            self.XC = XC

    def k_fold_cross_validation(self, X, K):        
        for k in xrange(K):
            training = [x for i, x in enumerate(X) if i % K != k]
            validation = [x for i, x in enumerate(X) if i % K == k]
            yield training, validation
        
    def getTVInds(self, nFolds=5):
        ''' Splits the data set into train and validation sets
        
            calculates:
                traininds: indices for the training set
                validinds: indices for the validation set
        '''
        nData = self.XC.shape[0]
        segs = np.floor(np.linspace(0, nData, nFolds+1)).astype(np.int)
        inds = np.arange(nData)
        np.random.shuffle(inds)
        traininds = []
        validinds = []
        for t, v in self.k_fold_cross_validation(inds, nFolds):
            traininds.append(t)
            validinds.append(v)
        self.traininds = np.asarray(traininds)
        self.validinds = np.asarray(validinds)
        
    def loadData(self, sigmoid_1st_layer=False):
        self.loadContours()
        if self.max_images != None:
            self.sampleContours()
        print "Cleaning contours..."
        self.cleanContours()
        print "Creating contour images..."
        #if os.path.isfile(os.path.join(self.data_dir, 'contimgs.pkl')):
        #    self.contimgs = pickle.load(file(os.path.join(self.data_dir, 'contimgs.pkl'), 'rb'))
        #else:
        print len(self.cxc)
        self.makeContourImages()
        print "Processing ultrasound images..."
        self.combineUltrasoundAndContourImages(sigmoid=sigmoid_1st_layer)


    def heat_map(self, makefig=True):
        self.loadContours()
        self.cleanContours()
        self.makeContourImages()
        heatmap = np.zeros((29,32))
        for i in range(len(self.contimgs)):
            scaled = imresize(self.contimgs[i], (29,32), interp='bicubic')
            heatmap += scaled
        self.heatmap = np.double(heatmap)/np.max(heatmap)
        if makefig:
            import pylab 
            import matplotlib.cm as cm
            pylab.imshow(heatmap,cm.gray)
            pylab.savefig('heatmap.jpg')
            pylab.clf()
            pylab.close()


if __name__ == "__main__":
    l = Loader("c:/Users/Jeff/aln0000JPG/olddiv/diverse-test/Subject1/")
    l.loadData()
    print np.max(l.XC), np.min(l.XC), np.mean(l.XC)


