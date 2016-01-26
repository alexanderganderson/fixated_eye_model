import os
import re
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf
import numpy as np
from scipy import interp
from pykalman import KalmanFilter


# Getting the Files


def get_mat_fns(data_dir='data_raw'):
    """
    Get all filenames (including path) in data_dir
    """
    mat_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.mat'):
                mat_files.append(os.path.join(root, f))
    print 'Mat Files Pre-Filtering: {}'.format(len(mat_files))
    return mat_files


def dm(m):
    return m[1:] - m[:-1]


def get_good_trials(mat_files, run_data, n_t=840, wsize=7):
    """
    Gets good trials:
    -- Searches filenames for subject ID to get pixel to arcmins conversion,
        otherwise, ignores that trial
    -- Trial is at frequency of n_t
    -- No zero acceleration for window > wsize

    Parameters
    ----------
    run_data : list (subject_id, pix_to_arcmin)
        List of subject_ids (search filename), and conversions
            for pixels to arcmins

    Returns
    -------
    trials : list of items (x, y, t, xp, yp)
        x, y : array, shape (n_t,)
            X position, Y position of eye trajectory in arcmin
        t : array, shape (n_t,)
            Time for each sample of the eye trajectory
        xp, yp : array, shape (n_t,)
            X position, Y position after smoothing outliers
    """
    trials = []
    id_not_found, wrong_size, zero_accel = 0, 0, 0
    for f in mat_files:
        # Select Certain Subjects
        pix_to_arcmin = None
        for subj_id, value in run_data:
            if re.search(subj_id, f):
                pix_to_arcmin = value
        if pix_to_arcmin is None:
            id_not_found += 1
            continue
        data = loadmat(f)
        x, y = data['frameshifts_strips_spline'].T / pix_to_arcmin
        x, y = x - x[0], y - y[0]
        t = data['timeaxis_secs'][:, 0]
        if x.size != n_t:
            wrong_size += 1
            continue
        if zero_accel_window(x, y, wsize=wsize):
            zero_accel += 1
            continue
        v = (dm(x) ** 2 + dm(y) ** 2) ** 0.5
        out = find_outliers_mad(v, 5) + 1
        xp = interp_outliers(t, x, out)
        yp = interp_outliers(t, y, out)
        trials.append((x, y, t, xp, yp))
    print 'Files Post Filtering: {}'.format(len(trials))
    print 'ID Not Found {} | Wrong Length {} | Zero Accel {}'.format(
        id_not_found, wrong_size, zero_accel)
    return trials


def find_outliers_mad(signal, thresh=5.):
    """
    Identify the outliers using the mean absolute deviation.
    signal : array, shape (n_t,)
    """
    diff = abs(signal - np.median(signal))
    return np.where(0.6745 * diff / (np.median(diff) + 1e-10) > thresh)[0]


def interp_outliers(t, x, out):
    """
    Interpolates a signal at given indices
    t : array, shape (n_t,)
        Times for signal
    x : array, shape (n_t,)
        Value of signal
    out : array, int
        indices corresponding to outliers
    """
    nout = np.delete(np.arange(x.size), out)  # Indices of non-outliers
    xp = x.copy()
    xp[out] = interp(t[out], t[nout], x[nout])
    return xp


def zero_accel_window(x, y, wsize=7, thresh=5e-10):
    """
    Return true if acceleration is zero for a window of size wsize

    Parameters
    ----------
    x, y: array, shape (n_t,)
        X, Y coordinates for path
    wsize : int
        The window size over which to search.
        7 is the number optimized by testing
    thresh : float
        Threshold for a small acceleration
    """
    eua = np.sqrt(dm(dm(x))**2 + dm(dm(y))**2)
    idx = np.where(eua < thresh)[0]
    # Look for windows where acceleration is below threshold
    if len(idx) < wsize:
        return False
    sums = np.convolve(np.ones(wsize), dm(idx) - 1, mode='valid')
    return len(np.where(sums == 0)[0]) > 0


# Eye trace models, fitting, and validating


def build_vdm(iv, dcv, ov, dt):
    """
    Builds KalmanFilter Object for Velocity Diffusion Model

    Parameters
    ----------
    iv : float
        Variance of initial velocity for x, y
    dcv : float
        Velocity diffusion constant
    ov : float
        Variance of measurement error.
    dt : float
        Timestep

    Returns
    -------
    xp, yp : array shape (n_t,)
        Kalman Smoother means
    """

    tv = dcv * dt
    adj = np.sqrt(1 - tv / iv)  # Keep Ev^2 constant
    return KalmanFilter(
        initial_state_mean=np.array([0, 0, 0, 0]),
        initial_state_covariance=np.array([[0, 0, 0, 0],
                                           [0, iv, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, iv]]),
        transition_matrices=np.array([[1, dt, 0, 0],
                                      [0, adj, 0, 0],
                                      [0, 0, 1, dt],
                                      [0, 0, 0, adj]]),
        transition_covariance=np.array([[0, 0, 0, 0],
                                        [0, tv, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, tv]]),
        observation_matrices=np.array([[1, 0, 0, 0],
                                       [0, 0, 1, 0]]),
        observation_covariance=np.array([[ov, 0],
                                         [0, ov]]))


def vdm_filter(trials, kf, n_t):
    """
    Use Velocity Diffusion Model to filter trials

    Parameters
    ----------
    trials : list (x, y, t, xp, yp)
        x, y : array, shape (n_t,)
            X, Y Coordinate of raw eye trace
        t : array, shape (n_t,)
            Time of each sample
        xp, yp : array, shape (n_t,)
            X, Y coordinate of eye trace filtered by MAD
    kf : KalmanFilter
        VDM model from build_vdm
    n_t : int
        Number of timesteps of data

    Returns
    -------
    x_, y_, t_, xp_, yp_ : array shape (n_trials, n_t)
        Data from list as matrices
    xf_, yf_, vxf_, vyf_ : array shape (n_trials, n_t)
        Kalman filter state means for position and velocity from VDM
    """
    (x_, y_, t_, xp_, yp_,
     xf_, yf_, vxf_, vyf_) = np.zeros((9, len(trials), n_t))
    for i, (x, y, t, xp, yp) in enumerate(trials):
        measurements = np.vstack((xp, yp)).T
        xf, vxf, yf, vyf = kf.smooth(measurements)[0].T

        x_[i], y_[i], t_[i], xp_[i], yp_[i] = x, y, t, xp, yp
        xf_[i], yf_[i], vxf_[i], vyf_[i] = xf, yf, vxf, vyf
    return x_, y_, t_, xp_, yp_, xf_, yf_, vxf_, vyf_


def build_dm(dc, ov, dt):
    """
    Build KalmanFilter Object for Diffusion Model

    Parameters
    ----------
    dc : float
        Diffusion constant
    ov : float
        Variance of measurement error.
    """
    var = dc * dt
    return KalmanFilter(
        initial_state_mean=np.array([0, 0]),
        initial_state_covariance=np.array([[0, 0], [0, 0]]),
        transition_matrices=np.array([[1, 0], [0, 1]]),
        transition_covariance=np.array([[var, 0], [0, var]]),
        observation_matrices=np.array([[1, 0], [0, 1]]),
        observation_covariance=np.array([[ov, 0], [0, ov]]))


def dm_filter(trials, kf, n_t):
    """
    Use Diffusion Model to filter trials

    Parameters
    ----------
    trials : list (x, y, t, xp, yp)
        x, y : array, shape (n_t,)
            X, Y Coordinate of raw eye trace
        t : array, shape (n_t,)
            Time of each sample
        xp, yp : array, shape (n_t,)
            X, Y coordinate of eye trace filtered by MAD
    kf : KalmanFilter
        DM model from build_dm
    n_t : int
        Number of timesteps of data

    Returns
    -------
    x_, y_, t_, xp_, yp_ : array shape (n_trials, n_t)
        Data from list as matrices
    xf_, yf_: array shape (n_trials, n_t)
        Kalman filter state means for position from DM
    """
    (x_, y_, t_, xp_, yp_,
     xf_, yf_) = np.zeros((7, len(trials), n_t))
    for i, (x, y, t, xp, yp) in enumerate(trials):
        measurements = np.array((xp, yp)).T
        xf, yf = kf.smooth(measurements)[0].T
        x_[i], y_[i], t_[i], xp_[i], yp_[i] = x, y, t, xp, yp
        xf_[i], yf_[i] = xf, yf
    return x_, y_, t_, xp_, yp_, xf_, yf_


def total_ll(trials, kf):
    s = 0.
    for x, y, _, xp, yp in trials:
        s += kf.loglikelihood(np.vstack((xp, yp)).T)
    return s / len(trials)


def cut_hist(data, eps=5e-4, bins=50):
    """
    Histogram of data in range [eps, 1 - eps]
    """
    hist, bin_edges = np.histogram(data, density=True, bins=bins)
    dx = bin_edges[1] - bin_edges[0]
    cum_probs = np.cumsum(hist) * dx
    idx = np.where((cum_probs > eps) * (cum_probs < 1 - eps))
    plt.bar(bin_edges[:-1][idx], hist[idx], width=dx)
    return hist[idx], bin_edges[idx]


def plot_hist_and_normal(data, std, bins=100, eps=5e-4):
    """Plot histogram of data and normal pdf"""
    vals, bins = cut_hist(data, bins=bins, eps=eps)
    p_bins = normpdf(bins, 0, std) + 1e-5
    plt.plot(bins, p_bins, c='g', linewidth=3)
    plt.yscale('log')


# Generic Plotting


def plot_single_trace(x, y, t, a=3.5):
    plt.scatter(x - x.mean(), y - y.mean(), c=t, alpha=0.75)
    plt.colorbar()
    plt.xlabel('x (arcmin)')
    plt.ylabel('y (arcmin)')
    plt.ylim([-a, a])
    plt.xlim([-a, a])


def plot_multiple_traces(x_, y_, t_, n=9, a=3.5, col=3):
    """
    Plot multiple traces

    Parameters
    ----------
    x_, y_: array, shape (n_trials, n_t)
        X,Y coordinates of traces to plot
    t_ : array, shape (n_trials, n_t)
        Time of each observation coloring purposes
    n : int
        Number of traces to plot
    a : float
        Plotting range is [-a, a] for x,y
    col : int
        Number of columns
    """
    row = (n + 2) / 3  # number of rows of images
    plt.figure(figsize=(4 * col, row * 3.5))
    for i, (x, y, t) in enumerate(zip(x_, y_, t_)):
        if i == n:
            break
        plt.subplot(row, col, i + 1)
        plot_single_trace(x, y, t, a=a)


# Post Processing

def resample(data0, dt0, dtp):
    """
    Resample sampled data to a new frequency

    Parameters
    ----------
    data0 : (n_trials, n_t0)
        Data matrix with second axis as time
    dt0 : float
        time between samples in original data
    dtp : float
        time between samples for output

    Returns
    -------
    datap : (n_trials, n_tp)
    """
    n_tr, n_t0 = data0.shape

    # Number of timesteps for new data
    n_tp = int(n_t0 * dt0 / dtp)
    t0 = np.arange(n_t0) * dt0
    datap = np.zeros((n_tr, n_tp))
    tp = np.arange(0, n_tp) * dtp
    for i, s0 in enumerate(data0):
        datap[i] = interp(tp, t0, s0)
    return datap


def generate_samples(kf, n_t, mode='dm', n_samp=9, alpha=0.5):
    """
    Generate Sample Trajectories

    Parameters
    ----------
    kf : KalmanFilter
        KF used to generate samples
    n_t : int
        Number of timesteps to sample data
    mode : str
        Either 'dm' or 'vdm'
    n_samp : int
        Number of sample trajectory to generate
    alpha : float
        Percent of observation variance to add back into smooth
            inferred path

    Returns
    -------
    xs_, ys_ : array, shape (n_samp, n_t)
        Sampled trajectories
    """
    if mode == 'dm':
        idx, idy = 0, 1
    elif mode == 'vdm':
        idx, idy = 0, 2
    xs_ = np.zeros((n_samp, n_t))
    ys_ = np.zeros((n_samp, n_t))
    for i in range(n_samp):
        (hid, obs) = kf.sample(n_t)
        xs_[i] = obs[:, 0] * (1 - alpha) + hid[:, idx] * alpha
        ys_[i] = obs[:, 1] * (1 - alpha) + hid[:, idy] * alpha
    return xs_, ys_

# Save


def save_traces(x_, y_, dt, data_desc='',
                output_dir='clean_data/',
                out_fn='paths.mat'):
    """
    Save traces

    Parameters
    ----------
    x_, y_ : array, shape (n_trials, n_t)
        Eye traces to save
    dt : float
        Timestep of the data
    data_desc : str
        String describing the data
    output_dir : str
    out_fun : str

    Returns
    -------
    d : dict
        Dictionary containing the data
        'paths' : array, shape (n_trials, n_t, 2)
            x_, y_ saved in an array
        'data_desc' : str
            data_desc
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    d = {}
    d['paths'] = np.array([x_, y_]).transpose(1, 2, 0)
    d['data_desc'] = data_desc
    d['DT'] = dt

    savemat(os.path.join(output_dir, out_fn), d)


if __name__ == '__main__':
    print 'Assumes that the trials are one second long at one frequency.'

    data_dir = 'data_raw'
    n_t = 840  # Choose 840Hz trials lasting for 1 second.

    run_data = [('20073', 420. / 60.)]  # , ('20014', 489. / 60.)]

    mat_files = get_mat_fns(data_dir=data_dir)
    trials = get_good_trials(mat_files, run_data, n_t=n_t)[0:50]

    iv, dcv, ov = 40., 50., 0.1
    kf = build_vdm(iv, dcv, ov, 1. / n_t)
    (x_, y_, t_, xp_, yp_,
     xf_, yf_, vxf_, vyf_) = vdm_filter(trials, kf, n_t)

    data_desc = (
        '20073L_10_9_2014 traces Filtered using VDM model' + '\n' +
        'VDM Parameters are iv = {:0.3f}, dcv = {:0.3f}, ov = {:0.3f}'.format(
            iv, dcv, ov) + '\n' +
        'Half of the noise is added back in after generating a smooth path')

    n_t_new = 1000.
    alpha = 0.5  # Restore some percent of variance from a smooth path
    x__, y__ = [resample(p_ * (1 - alpha) + f_ * alpha, 1./n_t, 1./n_t_new)
                for p_, f_ in [(xp_, xf_), (yp_, yf_)]]

    save_traces(x__, y__, 1./n_t_new, data_desc=data_desc)
