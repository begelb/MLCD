import torch 
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import itertools
from cycler import cycler
import matplotlib.pyplot as plt

''' filled_hist and stack_hist are taken from the following link: https://matplotlib.org/stable/gallery/lines_bars_and_markers/filled_step.html '''

def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                **kwargs):
    """
    Draw a histogram as a stepped patch.

    Parameters
    ----------
    ax : Axes
        The axes to plot to

    edges : array
        A length n+1 array giving the left edges of each bin and the
        right edge of the last bin.

    values : array
        A length n array of bin counts or values

    bottoms : float or array, optional
        A length n array of the bottom of the bars.  If None, zero is used.

    orientation : {'v', 'h'}
       Orientation of the histogram.  'v' (default) has
       the bars increasing in the positive y-direction.

    **kwargs
        Extra keyword arguments are passed through to `.fill_between`.

    Returns
    -------
    ret : PolyCollection
        Artist added to the Axes
    """
    print(orientation)
    if orientation not in 'hv':
        raise ValueError(f"orientation must be in {{'h', 'v'}} "
                         f"not {orientation}")

    kwargs.setdefault('step', 'post')
    kwargs.setdefault('alpha', 0.7)
    edges = np.asarray(edges)
    values = np.asarray(values)
    if len(edges) - 1 != len(values):
        raise ValueError(f'Must provide one more bin edge than value not: '
                         f'{len(edges)=} {len(values)=}')

    if bottoms is None:
        bottoms = 0
    bottoms = np.broadcast_to(bottoms, values.shape)

    values = np.append(values, values[-1])
    bottoms = np.append(bottoms, bottoms[-1])
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms,
                                **kwargs)
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms,
                               **kwargs)
    else:
        raise AssertionError("you should never be here")


def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
               hist_func=None, labels=None,
               plot_func=None, plot_kwargs=None):
    """
    Parameters
    ----------
    ax : axes.Axes
        The axes to add artists too

    stacked_data : array or Mapping
        A (M, N) shaped array.  The first dimension will be iterated over to
        compute histograms row-wise

    sty_cycle : Cycler or operable of dict
        Style to apply to each set

    bottoms : array, default: 0
        The initial positions of the bottoms.

    hist_func : callable, optional
        Must have signature `bin_vals, bin_edges = f(data)`.
        `bin_edges` expected to be one longer than `bin_vals`

    labels : list of str, optional
        The label for each set.

        If not given and stacked data is an array defaults to 'default set {n}'

        If *stacked_data* is a mapping, and *labels* is None, default to the
        keys.

        If *stacked_data* is a mapping and *labels* is given then only the
        columns listed will be plotted.

    plot_func : callable, optional
        Function to call to draw the histogram must have signature:

          ret = plot_func(ax, edges, top, bottoms=bottoms,
                          label=label, **kwargs)

    plot_kwargs : dict, optional
        Any extra keyword arguments to pass through to the plotting function.
        This will be the same for all calls to the plotting function and will
        override the values in *sty_cycle*.

    Returns
    -------
    arts : dict
        Dictionary of artists keyed on their labels
    """
    # deal with default binning function
    if hist_func is None:
        hist_func = np.histogram

    # deal with default plotting function
    if plot_func is None:
        plot_func = filled_hist

    # deal with default
    if plot_kwargs is None:
        plot_kwargs = {}
    print(plot_kwargs)
    try:
        l_keys = stacked_data.keys()
        label_data = True
        if labels is None:
            labels = l_keys

    except AttributeError:
        label_data = False
        if labels is None:
            labels = itertools.repeat(None)

    if label_data:
        loop_iter = enumerate((stacked_data[lab], lab, s)
                              for lab, s in zip(labels, sty_cycle))
    else:
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

    arts = {}
    for j, (data, label, sty) in loop_iter:
        if label is None:
            label = f'dflt set {j}'
        label = sty.pop('label', label)
        vals, edges = hist_func(data)
        if bottoms is None:
            bottoms = np.zeros_like(vals)
        top = bottoms + vals
        print(sty)
        sty.update(plot_kwargs)
        print(sty)
        ret = plot_func(ax, edges, top, bottoms=bottoms,
                        label=label, **sty)
        bottoms = top
        arts[label] = ret
    #ax.legend(fontsize=25)
    return arts

def plot_loss_histogram(loss_success_list, loss_failure_list, train_or_test_str, path_file, system, N, x_axis_label):
    loss_array = np.array([np.array(loss_success_list).transpose(), np.array(loss_failure_list).transpose()], dtype = object)

    # get the min and max of x and y
    if len(loss_success_list) > 0 and len(loss_failure_list) > 0:
        x_min = min(loss_success_list)
        x_max = max(loss_success_list)
        y_min = min(loss_failure_list)
        y_max = max(loss_failure_list)
        hist_min = x_min if x_min < y_min else y_min
        hist_max = x_max if x_max > y_max else y_max
    
    if len(loss_success_list) == 0:
        hist_min = min(loss_failure_list)
        hist_max = max(loss_failure_list)
    
    if len(loss_failure_list) == 0:
        hist_min = min(loss_success_list)
        hist_max = max(loss_success_list)

    # set up histogram function to fixed bins
    edges = np.linspace(hist_min, hist_max, 30, endpoint=True)
    hist_func = partial(np.histogram, bins=edges)

    # set up style cycles
    color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:2])
    label_cycle = cycler(label=['Successful trials', 'Failed trials'])
    hatch_cycle = cycler(hatch=['/', '.'])

    dict_data = dict(zip((c['label'] for c in label_cycle), loss_array))

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)

    stack_hist(ax, dict_data, color_cycle + label_cycle + hatch_cycle, hist_func=hist_func)

    ax.set_ylabel('Stacked frequency', fontsize = 25)
    ax.set_xlabel(x_axis_label, fontsize = 25)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    #if train_or_test_str == 'train':
    #    ax.set_title(f'System {system} (N = {N}): Train Loss Distribution for Successful and Unsuccessful Trials')
    #if train_or_test_str == 'test':
    #    ax.set_title(f'System {system} (N = {N}): Test Loss Distribution for Successful and Unsuccessful Trials')

    plt.savefig(path_file + '.png')
    plt.show()
    plt.close()
