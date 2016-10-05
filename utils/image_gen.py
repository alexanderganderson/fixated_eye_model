"""Simple class to generate different images."""

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.io import loadmat


class ImageGenerator:
    """Simple class to generate stimuli."""

    def __init__(self, l_i):
        """
        Initialize the image generator.

        Parameters
        ----------
        l_i - linear dimension of image
        """
        self.l_i = l_i
        self.reset_img()

    def reset_img(self):
        """Reset the image and image name."""
        self.img = np.zeros((self.l_i, self.l_i), dtype='float32')
        self.img_name = ''

    def dot(self):
        """Create a dot image."""
        self.img[self.l_i / 2, self.l_i / 2] = 1.

    def make_digit(self, mode='fixed'):
        """
        Make the image a MNIST digit.

        Parameters
        ----------
        mode - 'fixed' gives you a particular 0
             - 'random' gives you a random digit
        """
        if self.l_i != 14:
            raise ValueError('To create a digit, l_i = 14')

        try:
            data = loadmat('data/mnist_small.mat')
            images = data['IMAGES']
            labels = data['LABELS'][0]

            K, self.l_i, _ = images.shape

            if mode == 'fixed':
                # Chose a particular image that will work well
                k = 37  # zero
                k = 35  # two
            elif mode == 'random':
                k = np.random.randint(K)
            else:
                raise ValueError('unrecognized mode')

            self.reset_img()
            self.k = k
            self.img[:, :] = images[k]
            self.img_name = str(labels[k])

        except IOError, e:
            print e
            raise IOError(e)

    def make_gabor(self, x0=0., y0=0., f=None,
                   eta=np.pi / 4., phi=0., sig=None):
        """
        Make a Gabor.

        Parameters:
        x0, y0: float
            Gabor center relative to the center of the image
        f : float
            Frequency
        eta : float
            Frequency angle
        phi : float
            Phase
        img(x,y) = A * Exp( -((x-x0)^2 + (y-y0)^2)/sig ** 2) * Cos(kx + phi)
        kx = k (cos eta, sin eta) * (x, y)
        k = 2 * pi * f
        """
        if f is None:
            f = 2. / self.l_i
        if sig is None:
            sig = self.l_i / 3.

        tmp = np.arange(-self.l_i / 2, self.l_i / 2)
        X, Y = np.meshgrid(tmp, tmp)

        k = 2 * np.pi * f
        k1 = k * np.cos(eta)
        k2 = k * np.sin(eta)

        self.img = (np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / sig ** 2) *
                    np.sin(phi + k1 * X + k2 * Y))
        self.img_name = 'gabor'
        self.normalize()
        self.img = self.img.astype('float32')

    def make_e(self):
        """Make a one pixel thick E."""
        self.img[1, 2:-1] = 1
        self.img[self.l_i / 2, 2:-1] = 1
        self.img[-2, 2:-1] = 1
        self.img[1:-1, 2] = 1
        self.img_name = 'E'

    def make_big_e(self):
        """Make a two pixel thick E centered on the image."""
        l = self.l_i
        self.img[l/2-1:l/2+1, l/2-5:l/2+5] = 1
        self.img[l/2-5:l/2-3, l/2-5:l/2+5] = 1
        self.img[l/2+3:l/2+5, l/2-5:l/2+5] = 1
        self.img[l/2-5:l/2+5, l/2-5:l/2-3] = 1
        self.img_name = 'bigE'

    def random(self):
        """Create a white noise image."""
        self.img[:, :] = np.random.random(
            (self.l_i, self.l_i)).astype('float32')
        self.img_name = 'white_noise'

    def make_t(self):
        """Create a one pixel width T."""
        self.img[1, 1:-1] = 1
        self.img[2:-1, self.l_i / 2] = 1
        self.img_name = 'T'

    def smooth(self, a=3, sig=0.1):
        """Blur the image."""
        x = np.arange(-a, a + 1).astype('float32')
        y = x
        xg, yg = np.meshgrid(x, y)
        f = np.exp(-(xg ** 2 + yg ** 2) / sig ** 2).astype('float32')
        self.img = convolve2d(self.img, f, mode='same')

    def normalize(self, m1=0., m2=1.):
        """Normalize the image to have min = m1, max = m2."""
        self.img = self.img - self.img.min()
        self.img = self.img / self.img.max()

        self.img = self.img * (m2 - m1) + m1

    def variance_normalize(self):
        """Normalize the image to have sum of squares = 1."""
        self.img = self.img / np.sqrt(np.sum(self.img ** 2))

    def plot(self):
        """Plot the image."""
        plt.imshow(self.img,
                   interpolation='nearest',
                   cmap=plt.cm.gray_r)
        plt.colorbar()
        plt.title('Image')
        plt.show()


def rotated_e():
    """
    Create E's for a tumbling E task.

    Returns
    -------
    img : array, shape (4, 5, 5)
        A collection of four 5 by 5 images of all rotations of an E
    """
    x = np.zeros((5, 5))
    x[:, 0] = 1.
    y = np.zeros((5, 5))
    y[:, 2] = 1.
    z = np.zeros((5, 5))
    z[:, 4] = 1.
    a = np.zeros((5, 5))
    a[0, :] = 1.
    b = np.zeros((5, 5))
    b[2, :] = 1.
    c = np.zeros((5, 5))
    c[4, :] = 1.

    img = np.zeros((4, 5, 5))
    img[0] = x + y + z + a
    img[1] = x + y + z + c
    img[2] = a + b + c + x
    img[3] = a + b + c + z
    img[img > 0] = 1.

    return img.astype('float32')


def show_es():
    """Show E's from tumbling E creator."""
    img = rotated_e()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img[i], cmap=plt.cm.gray, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    ig = ImageGenerator(14)
