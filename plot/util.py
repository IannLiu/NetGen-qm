from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
import os
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
'''@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)'''

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def plot_simulations(results: pd.DataFrame,
                     haxis: str,
                     vaxis: List[str] = None,
                     colors: List[str] = None,
                     max_species_num: int = None,
                     save_path: str = None
                     ):
    """
    Plot simulation results
    
    Args:
        results: simulation results, including time points and simulation results
        haxis: the horizontal axis name
        vaxis: the vertical axis name
        colors: colors of curves
        max_species_num: show top n species curves
        save_path: save path

    Returns:
    """
    x = results[haxis].values
    y = results.drop(columns=[haxis]) if vaxis is None else results[[vaxis]]
    y_len = int(y.shape[0] - 1)
    y.sort_values(by=y_len, axis=1, inplace=True)
    fig = plt.figure()
    if max_species_num is not None:
        data = y.iloc[0: max_species_num].values
        labels = list(y.columns[0: max_species_num])
    else:
        data = y.values
        labels = list(y.columns)
        print(labels)
    if colors is not None:
        assert len(colors) == max_species_num, 'The length of color list have to be equal to the the max_species_num'
        plt.plot(x, data, colors=colors, label=labels)
    else:
        plt.plot(x, data, label=labels)
    plt.legend(loc='best', fontsize=8)
    return y

