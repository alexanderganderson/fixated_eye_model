# Image Generator
#
# Simple class that generates a few different images

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.io import loadmat

class ImageGenerator:
    def __init__(self, _L_I):
        """
        L_I - linear dimension of image
        """
        self.L_I = _L_I
        self.reset_img()
        
    def reset_img(self):
        self.img = np.zeros((self.L_I, self.L_I), dtype = 'float32')
        
    def dot(self):
        self.img[self.L_I / 2, self.L_I / 2] = 1.
    
    
    def make_digit(self):
        try:
            data = loadmat('data/mnist_small1.mat')
            IMAGES = data['IMAGES']
            K, self.L_I, _ = IMAGES.shape
            #k = np.random.randint(K)
            k = 37 # Chose a particular image that will work well
            self.reset_img()
            self.img[:, :] = IMAGES[k]
        except IOError, e:
            print e
            print 'MNIST file not found, making big E instead'
            self.make_big_E()
    
    def make_gabor(self, x0 = 0., y0 = 0., k = None,
                   eta = np.pi / 4., phi = 0., sig = None):
        """
        Make a Gabor 
        x0, y0 - gabor center relative to the center of the image
        k = frequency amplitude
        eta - frequency angle
        phi - phase
        img(x,y) = A * Exp( -((x-x0)^2 + (y-y0)^2)/sig ** 2) * Cos(kx + phi)
        kx = k (cos eta, sin eta) * (x, y)
        """
        
        if (k == None):
            k = 4. * np.pi / self.L_I
        if (sig == None):
            sig = self.L_I / 3.
        
        
        tmp = np.arange(-self.L_I / 2, self.L_I / 2)
        X, Y = np.meshgrid(tmp, tmp)
        
        k1 = k * np.cos(eta)
        k2 = k * np.sin(eta)
        
        self.img = (np.exp( -((X - x0) ** 2 + (Y - y0) ** 2) / sig ** 2 ) *
                    np.sin(phi + k1 * X + k2 * Y))
        
        self.normalize()    
    
    def make_E(self):
        self.img[1, 2:-1] = 1
        self.img[self.L_I / 2, 2:-1] = 1
        self.img[-2, 2:-1] = 1
        self.img[1:-1, 2] = 1
        
    def make_big_E(self):
        self.img[1:3, 2:-2] = 1
        self.img[self.L_I / 2 - 1: self.L_I/2 + 1, 2:-2] = 1
        self.img[-4:-2, 2:-2] = 1
        self.img[1:-2, 2:4] = 1
        
        
    def random(self):
        self.img[:, :] = np.random.random(
            (self.L_I, self.L_I)).astype('float32')
            
    def make_T(self):
        self.img[1, 1:-1] = 1
        self.img[2:-1, self.L_I / 2] = 1
        
    def smooth(self, a = 3, sig = 0.1):
        X = np.arange(-a, a+1).astype('float32')
        Y = X
        Xg, Yg = np.meshgrid(X, Y)
        F = np.exp(-(Xg ** 2 + Yg ** 2) / sig ** 2).astype('float32')
        self.img = convolve2d(self.img, F, mode = 'same')
        
    def normalize(self, m1 = 0., m2 = 1.):
        """
        Normalizes the image to have min = m1, max = m2
        """
        self.img = self.img - self.img.min()
        self.img = self.img / self.img.max()
        
        self.img = self.img * (m2 - m1) + m1
        
        
    def variance_normalize(self):
        """
        Normalizes the image to have sum of squares = 1
        """
        
        self.img = self.img / np.sqrt(np.sum(self.img ** 2))
        
    def plot(self):
        plt.imshow(self.img, 
                   #interpolation = 'nearest', 
                   cmap = plt.cm.gray_r)
        plt.colorbar()
        plt.title('Image')
        plt.show()

if __name__ == '__main__':
    ig = ImageGenerator(14)