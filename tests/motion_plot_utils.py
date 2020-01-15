import os
import numpy as np

from collections import OrderedDict
import pandas as pd

from context import src, utils
from src.analyzer import DataAnalyzer, plot_fill_between
from utils.plot_utils import label_subplot, equalize_y_axes

EXP = 'Experiment'
NOM = 'Diffusion'

def nice_print(lst):
    l1 = []
    for l in lst:
        if isinstance(l, str):
            l1.append(l)
        elif isinstance(l, float):
            l1.append('{:6.2f}'.format(l))
        elif isinstance(l, int):
            l1.append('{:02}'.format(l))
        else:
            l1.append(str(l))
    return ' | '.join(l1)


def plot_results(ax, grouped):
    """Plot SNR vs. Time for the different inference modes."""
    label_ = []
    rect_ = []
    for name, group in grouped:
        gen_mode = name['gen_mode']
        inf_mode = name['inference_mode']
        label = ({EXP: 'S:M', NOM: 'S:NM'}[gen_mode] + ' | ' + 
                 {'EM': 'D:EM ', 'NoMotion': 'D:NM'}[inf_mode])
        plot_fill_between(ax, t, group[list(t)], label=label, confidence_interval=True)
        c = ax.get_lines()[-1].get_color()
        rect_.append(Rectangle((0, 0), 1, 1, fc=c, hatch=None, linewidth=0))
        label_.append(label)
    ax.legend(rect_, label_, loc='upper left', prop={'size': 7}, labelspacing=0.2)
    

def tuning_plot(
    ax, grouped, tuning_key, x_label='', y_label='SNR (t=700 ms)', 
    x_normalizer=1., log_yscale=False, t=None, loc='best',
):
    item_ = []

    for name, group in grouped:
        snrs = group[list(t)[-1]].values
        mean = snrs.mean()
        std = snrs.std() / np.sqrt(len(snrs))
        item = {
            tuning_key: name[tuning_key], 
            'gen_mode': name['gen_mode'], 
            'snr': snrs,
            'snr_mean': mean,
            'snr_std': std,
        }
        if tuning_key == 'ds' and name[tuning_key] > 0.7:
            continue
        item_ += [item]
    df = pd.DataFrame(item_)

    for name, group in df.groupby('gen_mode'):
        label = {EXP:'Motion', NOM:'No Motion'}[name]
        color = {EXP:'b', NOM: 'r'}[name]
        ax.errorbar(
            group[tuning_key] / x_normalizer, 
            group['snr_mean'], yerr=group['snr_std'], 
            label=label, color=color
        )
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if log_yscale:
        ax.set_yscale('log')

    ax.legend(loc=loc, prop={'size': 7}, labelspacing=0.2)
    
    
def collect_analyzers(output_dir_list):
    output_dir_list = [os.path.join('../output', folder) for folder in output_dir_list]
    data_fns = []
    for output_dir in output_dir_list: 
        data_fns.extend(
            [os.path.join(output_dir, fn) 
                   for fn in os.listdir(output_dir) 
                   if fn.endswith('.h5')]
            )
    data_fns.sort()

    print len(data_fns)
    da_ = map(DataAnalyzer.fromfilename, data_fns)

    for da in da_:
        da.s_range = 'pos'
    return data_fns, da_

def groupby(data, grouping_columns):
    """
    Groupby that has group names that are a dictionary.
    """
    grouped = data.groupby(grouping_columns)

    lst = []
    for name, group in grouped:
        if len(grouping_columns) == 1:
            name = (name,)
        name1 = OrderedDict(zip(grouping_columns, name))
        lst.append((name1, group))
    return lst

def analyzers_to_dataframe(da_):
    out_ = []
    for da in da_:
        out = {}
        out['inference_mode'] = da.data['EM_data/mode']
        out['gen_mode'] = da.data['motion_gen']['mode']   # True eye movements or no eye movements
        try:
            out['gen_dc'] = da.data['motion_gen']['dc']
        except KeyError:
            out['gen_dc'] = 0.0
        try:
            out['drop_prob'] = da.data['drop_prob']
        except KeyError:
            out['drop_prob'] = 0.0
        out['scaling_factor'] = da.data['motion_gen'].get('scaling_factor', 1.0)
        out['prior_dc'] = da.data['motion_prior']['dc']
        out['ds'] = da.data['ds'] # Image size
        for t, snr_t in zip(da.time_list(), da.snr_list()):
            out[t] = snr_t

        out_.append(out)
    data = pd.DataFrame(out_)
    return data