"""Sparse Coding Class."""

from itertools import product

import numpy as np
from scipy.linalg import eigh
import cPickle as pkl
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from fista import fista_updates, ista_updates

import sys
sys.path.append('..')
from utils.rf_plot import show_fields

def get_effective_dimensionality(data, var_thresh=0.9):
    """
    Get the effective dimensionality of a dataset using PCA.

    Parameters
    ----------
    data : array, shape (n_images, n_features)
        Data to be analyzed.
    var_thresh : float
        Percent of total variance to cut the dimensionality.

    Returns
    -------
    d_eff : int
        Effective dimensionality of the data.
    """
    IM_cov = np.cov(data.T)
    evals, evecs = eigh(IM_cov)
    cs = np.cumsum(evals[::-1])
    d_eff = np.argmax(cs > cs[-1] * var_thresh) # Effective Dim of data
    return d_eff


def get_pca_basis(data, var_thresh=0.9):
    """
    Get the effective dimensionality of a dataset using PCA.

    Parameters
    ----------
    data : array, shape (n_images, n_features)
        Data to be analyzed.
    var_thresh : float
        Percent of total variance to cut the dimensionality.

    Returns
    -------
    d: array, shape (d_eff, n_features)
        Basis that captures 90 percent of the variance in the data
    """
    IM_cov = np.cov(data.T)
    evals, evecs = eigh(IM_cov)
    cs = np.cumsum(evals[::-1])
    d_eff = np.argmax(cs > cs[-1] * var_thresh) # Effective Dim of data
    return evecs.T[::-1][0:d_eff]


def _build_base_masks(l_img):
    itr = product([0, l_img/4, l_img/2],
                  [0, l_img/4, l_img/2])
    itr = [i for i in itr]
    masks = np.zeros((1 + len(itr), l_img, l_img))
    for i, (u, v) in enumerate(itr):
        masks[i,
              u:u + l_img / 2,
              v:v + l_img / 2] = 1
        masks[-1] = 1
    return masks

def sparsifying_mask(l_img, n_sp):
    base_masks = _build_base_masks(l_img)
    k, _, _  = base_masks.shape
    repl = int(n_sp / k)
    masks = np.ones((n_sp, l_img, l_img))
    for i, base_mask in enumerate(base_masks):
        masks[i * repl:(i+1) * repl] = base_mask
    return masks

class SparseCoder(object):
    """
    Create a sparse code on a set of data.
    """

    def __init__(self, data, n_sp, n_bat, alpha, d_scale=1., pos_only=False,
                 sparsify=False):
        """

        Parameters
        ----------
        data : np.array, shape (n_imgs, n_pix)
            Data to train on.
        n_sp : int
            Number of sparse coefficients
        n_bat : int
            Batch size during learning
        alpha : float
            Sparsity coefficient
        pos_only : bool
            True if we do positive only learning
        sparsify: bool
            If True, generate a mask to sparsify the dictionary during
            learning.
        """

        self.n_imgs = data.shape[0]
        self.n_bat = n_bat
        self.d_scale = d_scale
        self.sparsify = sparsify
        self.tc = TheanoBackend(data=data, alpha=alpha, d_scale=d_scale,
                                pos_only=pos_only, n_bat=n_bat, n_sp=n_sp,
                                sparsify=sparsify)

    def save(self, path):
        """

        Parameters
        ----------
        path : str
            Path to save pkl file.
        """
        d = {'n_sp': self.tc.n_sp, 'n_bat': self.n_bat, 'alpha': self.tc.alpha,
             'd_scale': self.d_scale,
             'pos_only': self.tc.pos_only, 'D': self.tc.t_D.get_value(),
             'sparsify': self.sparsify}
        with open(path, 'wb') as fn:
            pkl.dump(d, fn)

    @classmethod
    def restore(cls, data, path, n_bat=None):
        with open(path, 'rb') as fn:
            d = pkl.load(fn)
        if n_bat is None:
            n_bat = d['n_bat']
        sparse_coder = cls(
            data, n_sp=d['n_sp'], n_bat=n_bat, alpha=d['alpha'],
            d_scale=d['d_scale'], pos_only=d['pos_only'], sparsify=d['sparsify'])

        sparse_coder.tc.t_D.set_value(d['D'])
        return sparse_coder

    def train(self, n_itr, eta, cost_list, n_g_itr=200, show_costs=True):
        """
        Code to train a sparse coding dictionary.

        Parameters
        ----------
        n_itr : int
            Number of iterations, a new batch of images for each
        eta : float
            Step size for dictionary updates.

        cost_list - list to which the cost at each iteration will be appended
        n_g_itr : int
            Number of gradient steps in FISTA
        show_costs - If true, print out costs every N_itr/10 iterations

        Returns
        -------
        i_idx - indices corresponding to the most recent image batch,
            to be used later in visualizations
        """
        if show_costs:
            print 'Iteration, E, E_rec, E_sp, SNR'

        for i in range(n_itr):
            i_idx = np.random.randint(self.n_imgs, size=self.n_bat).astype('int32')
            costs = self._run_train_step(i_idx=i_idx, eta=eta, n_g_itr=n_g_itr)
            E, E_rec, E_sp, SNR = costs
            cost_list.append(E)
            show_costs_now = ((i + 1) % (1 + n_itr / 100) == 0) and show_costs
            if show_costs_now:
                print i, E, E_rec, E_sp, SNR
        return i_idx


    def _run_train_step(self, i_idx, eta, n_g_itr, show_costs=False):
        """
        Run one training step.

        Parameters
        ----------

        """
        self.tc.reset_fista_variables()
        l = self.tc.calculate_fista_l(self.tc.t_D.get_value())
        for i in range(n_g_itr):
            E, E_rec, E_sp, SNR = self.tc.fista_step(i_idx, l)
            if show_costs:
                print i, E, E_rec, E_sp, SNR
        E, E_rec, E_sp, SNR = self.tc.costs(i_idx)
        self.tc.dictionary_learning_step(eta, self.d_scale, i_idx)
        return E, E_rec, E_sp, SNR

    def plot_example(self, q, i_idx, L_pat, ax=None):
        A = self.tc.get_coefficients()
        D = self.tc.get_dictionary()
        Ih = np.dot(A, D)
        I = self.tc.t_DATA.get_value()[i_idx]
        A = A[q]
        Ih = Ih[q]
        I = I[q]
        n_sp = D.shape[0]

        snr = (I ** 2).sum() / ((I - Ih) ** 2).sum()

        vmin = I.min()
        vmax = I.max()

        if ax is None:
            ax = plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.title('Reconstruction: SNR: {:.2f}'.format(snr))
        plt.imshow(Ih.reshape(L_pat, L_pat),
                   interpolation='nearest',
                   cmap = plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.title('Orignal Image')
        plt.imshow(I.reshape(L_pat, L_pat),
                   interpolation='nearest',
                   cmap = plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.subplot(2, 2, 3)

        plt.hist(A, bins=40)

        sort_idx = np.argsort(np.abs(A))[::-1]
        n_active = np.sum(np.abs(A) > 0.2)
        if n_active == 0:
            return
        active_idx = sort_idx[0:n_active]

        plt.title('Histogram of sparse Coefficients: \n Percentage of active coefficients {:.0f}%'.format(
                100. * n_active / n_sp))
        plt.xlabel('Coefficient Activity')

        plt.subplot(2, 2, 4)
        show_fields(D[active_idx] *
                    A[active_idx][:, np.newaxis],
                    cmap = plt.cm.gray, pos_only = False)
        plt.title('Active Dictionary Elements \n Scaled by their activations')




class TheanoBackend(object):
    """
    """
    def __init__(self, data, alpha, d_scale, pos_only, n_bat, n_sp, sparsify):
        """

        Parameters
        ----------
        data : array, shape (n_imgs, n_pix) float32
            Images to train the data.
        pos_only : bool

        n_bat : int
            Number if images in a training batch
        n_sp : int

        """
        self.alpha = alpha
        self.pos_only = pos_only
        self.n_bat = n_bat
        self.n_sp = n_sp
        _, n_pix = data.shape

        def threshold(t_X, pos_only=pos_only):
            """
            Threshold function. Does nothing if not positive only
            """
            if pos_only:
                return T.switch(t_X > 0., t_X, -0.00 * t_X )
            else:
                return t_X

        t_L = T.scalar('L')  # FISTA L
        t_Eta = T.scalar('eta')  # Dictionary update stepsize
        t_D_scale = T.scalar('D_scale')  # Target scale of dictionary

        def row_norm(t_X, t_scale=t_D_scale):
            """
            Returns row normalized version of a theano matrix t_X
            the rows have a norm of t_std
            """
            eps = 0.000001
            return t_scale * t_X / T.sqrt(eps + T.sum(t_X ** 2, axis = 1)).dimshuffle(0, 'x')


        D = np.random.uniform(size=(n_sp, n_pix)) # Randomly initialize the dictionary
        D = d_scale * D / np.sqrt(np.sum(D ** 2, axis=1, keepdims=True)) # Normalize dictionary elements
        t_D = theano.shared(D.astype('float32'), 'D')


        t_DATA = theano.shared(data, 'I')
        t_A = theano.shared(np.zeros((n_bat, n_sp), dtype='float32'), 'A')
        t_Alpha = theano.shared(np.array(alpha, dtype='float32'), 'Alpha')  # Sparsity weight
        t_I_idx = T.ivector('I_idx') # Indicies into t_I to select a batch of images
        t_I = t_DATA[t_I_idx]

        t_E_rec = T.sum((t_I - T.dot(t_A, t_D)) ** 2); t_E_rec.name = 'E_rec'
        t_E_sp = t_Alpha * T.sum(T.abs_(t_A)); t_E_sp.name = 'E_sp'
        t_E = t_E_rec + t_E_sp; t_E.name = 'E'

        eps = 0.000001
        t_SNR = T.mean(
            T.sum(t_I ** 2, axis = 1) /
            (eps + T.sum((t_I - T.dot(t_A, t_D)) ** 2, axis = 1)))

        costs = theano.function(inputs = [t_I_idx],
                                outputs = [t_E, t_E_rec, t_E_sp, t_SNR])

        t_gED = T.grad(t_E_rec, t_D)

        if sparsify is True:
            d_mask = sparsifying_mask(l_img=int(np.sqrt(n_pix)), n_sp=n_sp)
            d_mask = d_mask.reshape(d_mask.shape[0], -1)
            t_D_mask = theano.shared(d_mask.astype('float32'), 'd_mask')
            t_D_update = row_norm(t_D_mask * threshold(t_D - t_Eta * row_norm(t_gED)))
        else:
            t_D_update = row_norm(threshold(t_D - t_Eta * row_norm(t_gED)))

        dictionary_learning_step = theano.function(
            inputs = [t_Eta, t_D_scale, t_I_idx],
            outputs = [t_E, t_E_rec, t_E_sp],
            updates = [(t_D, t_D_update)])


        fist_updates = fista_updates(t_A, t_E_rec, t_Alpha, t_L, pos_only=pos_only)
        _, t_fista_X, t_T = fist_updates.keys()

        fista_step = theano.function(inputs = [t_L, t_I_idx],
                                     outputs = [t_E, t_E_rec, t_E_sp, t_SNR],
                                     updates = fist_updates)
        test_grad = theano.function(inputs=[t_I_idx],
                                    #  outputs=[T.grad(t_E_rec, t_A)])
                                    outputs=[t_E_rec,
                                             T.mean((T.grad(t_E_rec, t_A)) ** 2)])
        self.test_grad = test_grad

        self.t_A = t_A
        self.t_fista_X = t_fista_X
        self.t_T = t_T

        self.t_D = t_D


        self.t_DATA = t_DATA

        self._costs = costs
        self._dictionary_learning_step = dictionary_learning_step
        self._fista_step = fista_step

    def dictionary_learning_step(self, eta, d_scale, i_idx):
        """

        Parameters
        ----------
        alpha : float

        """
        self._dictionary_learning_step(eta, d_scale, i_idx)

    def fista_step(self, i_idx, l):
        return self._fista_step(l, i_idx)

    def costs(self, i_idx):
        return self._costs(i_idx)

    @staticmethod
    def calculate_fista_l(D):
        """
        Calculates the 'L' constant for FISTA for the dictionary
        """
        n_sp, _ = D.shape
        try:
            L = (2 * eigh(np.dot(D, D.T), eigvals_only=True, eigvals=(n_sp - 1, n_sp - 1))[0]).astype('float32')
        except ValueError:
            # FIXME: d_std unclear
            d_scale = np.sqrt(np.sum(D ** 2, axis=1).mean())
            L = (2 * d_scale ** 2 * n_sp).astype('float32') # Upper bound on largest eigenvalue
        return L

    def reset_fista_variables(self):
        """
        Resets fista variables
        """
        A0 = np.zeros_like(self.t_A.get_value()).astype(theano.config.floatX)
        self.t_A.set_value(A0)
        self.t_fista_X.set_value(A0)
        self.t_T.set_value(np.array([1.]).astype(theano.config.floatX))

    def get_dictionary(self):
        return self.t_D.get_value()

    def get_coefficients(self):
        return self.t_A.get_value()


