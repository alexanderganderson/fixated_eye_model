# Image Generator
#
# Simple class that generates a few different images

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


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
        
    def normalize(self):
        """
        Normalizes the image to have min 0, max 1
        """
        self.img = self.img - self.img.min()
        self.img = self.img / self.img.max()
        
    def plot(self):
        plt.imshow(self.img, 
                   interpolation = 'nearest', 
                   cmap = plt.cm.gray_r)
        plt.colorbar()
        plt.title('Image')
        plt.show()