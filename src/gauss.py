"""Impelement Gaussian Model."""

import numpy as np
import tensorflow as tf
import utils.particle_filter as pf




class EMGauss(object):
    """Produce spikes and infer underlying causes."""

    def __init__(self, l_i, motion_gen, motion_prior, n_t,
                 ds, de, save_mode, n_itr, n_p, print_mode):

        self.tb = TFBackend()

        self.pf = self.init_particle_filter()
        pass

    def gen_data(self, s_gen, pg=None):
        pass

    @staticmethod
    def init_pix_rf_centers():
        pass

    @staticmethod
    def init_particle_filter():
        pass

    def reset(self):
        pass

    def _run_e(self, m, t0, tf):
        while self.pf.t < t:
            self.pf.advance(Y=m)

        xr = self.pf.XS[t0:tf, :, 0].transpose()
        yr = self.pf.XS[t0:tf, :, 1].transpose()
        w0 = self.pf.WS[t0:tf, :].transpose()
        return xr, yr, w0

    def _run_m(self, a, b, m, w):
        a0, b0 = self.tb.get_ab(xr=xr, yr=yr, w=w, m=m)
        a = a + a0
        b = b + b0
        s =  np.linalg.solve(a, b)
        return a, b, s

    def run_em(self, m, n_passes, n_itr):
        a = np.zeros((n_pix, n_pix))
        b = np.zeros((n_pix,))
        s = np.zeros((n_pix,))

        for i in range(n_passes):
            self.pf.reset()
            a[:] = 0
            b[:] = 0
            for u in range(n_itr):
                t0 = self.n_t * u / n_itr
                tf = self.n_t * (u + 1) / n_itr

                m0 = m[:, t0:tf]
                xr0, yr0, w0 = self._run_e(m=m0, t0=t0, tf=tf)
                a, b, s = self._run_m(a, b, m0, w0)

    def _build_param_dict(self):
        pass

    def save(self):
        pass




def _get_t_matrix(t_xs, t_ys, t_xe, t_ye, t_xr, t_yr, t_var):
    """
    Get the 'T' matrix connecting pixels and receptive fields.

    Parameters
    ----------
    t_xs, t_ys : tf.Variable, shape (n_pix,)
        X, Y coordinates of image pixels.
    t_xe, t_ye : tf.Variable, shape (n_sensors,)
        X, Y coordinates of image sensors.
    t_xr, t_yr : tf.Variable, shape (n_p, n_t)
        X, Y coordinates of translations
    t_var : tf.Variable, shape (1,)
        Combined variances of pixel blurring and measuremnt RF size.

    Returns
    -------
    t_t, tf.Variable, shape (n_pix, n_sensors, n_translations)
        Matrix connecting pixels and samples.
    """

    c = tf.constant(1, dtype='float32', shape=(1,))
    t_xs, t_ys = [tf.einsum('i,j,p,t->ijpt', u, c, c, c) for u in [t_xs, t_ys]]

    t_xe, t_ye = [tf.einsum('i,j,p,t->ijpt', c, u, c, c) for u in [t_xe, t_ye]]

    t_xr, t_yr = [tf.einsum('i,j,pt->ijpt', c, c, u)  for u in [t_xr, t_yr]]

    t_d2 = (t_xs - t_xe - t_xr) ** 2 + (t_ys - t_ye - t_yr) ** 2
    PI = tf.constant(np.pi, dtype='float32')
    t_t = tf.exp(- t_d2 / (2 * t_Var)) / (2 * PI * t_Var)
    return t_t




def _calc_a(t_w, t_t):
    """
    Get A matrix

    Parameters
    ----------
    t_w : tf.Tensor, shape (n_p, n_t)
        Particle filter weights
    t_t : tf.Tensor, shape (n_pix, n_sensors, n_p, n_t)
        Tensor connecting pixels and samples

    Returns
    -------
    t_a : tf.Tensor, shape (n_pix, n_pix)
        Inverse covariance of image estimate.
    """
    return tf.einsum('pt,xjpt,yjpt->xy', t_w, t_t, t_t)




def _calc_b(t_w, t_m, t_t):
    """
    Get B vector

    Parameters
    ----------
    t_w : tf.Tensor, shape (n_p, n_t)
        Particle filter weights
    t_m : tf.Tensor, shape (n_receptors, n_t)
        Measurements
    t_t : tf.Tensor, shape (n_pix, n_sensors, n_p, n_t)

    Returns
    -------
    t_b : tf.Tensor, shape (n_pix,)

    """
    return tf.einsum('pt,jt,ijpt->i', t_w, t_m, t_t)



def _calc_batched_e(t_m, t_s, t_t, t_sig_m):
    """
    Calculate log p(M|X, S) batched over X.

    Parameters
    ----------
    t_m : tf.Tensor, shape (n_sensors, n_t)
        Measurements
    t_s : tf.Tensor, shape (n_pixels,)
        Pixels of image estimate
    t_t : tf.Tensor, shape (n_pixels, n_sensors, n_particles, n_t)
        Image to measurement matrix
    t_var_m : tf.Tensor, shape (n_sensors,)
        Variance of each measurement device

    Returns
    -------
    t_e : tf.Tensor, shape (n_particles, n_t)
        Energy batched over positions.
    """
    c = tf.constant(1, dtype='float32', shape=(1,))
    t_s = tf.einsum('i,j,p,t->ijpt', t_s, c, c, c)
    t_m0 = tf.einsum('ijpt,ijpt->jpt', t_s, t_t)
    t_m = tf.einsum('p,jt->jpt', c, t_m)
    tf.einsum('jpt->pt', (t_m - t_m0) ** 2






class TFBackend(object):
    """Runs core operations on GPU."""

    def __init__(self, xs, ys, xe, ye, var_s, var_m, SMIN=0., SMAX=1.):

        # Simulation constants
        t_xs, t_ys, t_xe, t_ye, t_var_s = [
            tf.Variable(u, name=name) for u, name in
            zip([xs, ys, xe, ye, var_s], ['xs', 'ys', 'xe', 'ye', 'var_s'])]

        t_xr, t_yr = [tf.placeholder('float32', shape=(None,), name=name),
                      for name in ['xr', 'yr']]

        # i, j, bt
        t_t = _get_t_matrix(t_xs, t_ys, t_xe, t_ye, t_xr, t_yr, t_var):

        # Generate measurements
        t_s_gen = tf.placeholder('float32', shape=(n_pix,), name='s_gen')
        t_s_genp = tf.expand_dims(tf.expand_dims(t_s_gen, 1), 1)

        t_m_gen = tf.reduce_sum(t_s_genp * t_t, axis=0)  # j, t

        # Generate inference equations

        t_w = tf.placeholder('float32', shape=(None, None), name='w')  # p, t
        t_m = tf.placeholder('float32', shape=(None, None), name='m')  # j, t


        t_a = _calc_a(t_w, t_t)
        t_b = _calc_b(t_w, t_m, t_t)

        # Batched energy for particle filter
        t_e = _calc_batched_e(t_m, t_s_gen, t_t)
        tf.einsum('i,ijpt->jpt', s_gen, t_t)

        t_e = tf.einsum('pjt->jt', (t_m - t_t * t_x) ** 2)

        init_op = tf.initialize_all_variables()
        self.sess = tf.Session()
        with self.sess.as_default():
            init_op.run()


    def get_ab(xr, yr, w, m, sess=None):
        if sess is None:
            sess = self.sess

        feed_dict = {
            t_xr: xr,
            t_yr: yr,
            t_w: w,
            t_m: m}
        with sess.as_default():
            a, b = sess.run(t_a, t_b, feed_dict=feed_dict)



