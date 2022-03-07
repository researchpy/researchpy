import pandas
import numpy
import scipy.stats





def count(d):
    """

        Counts the number of non-missing observations.

    """

    return numpy.count_nonzero(~numpy.isnan(d))



def nanvar(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The variance of the non-missing data passed; calculated as numpy.nanvar(d, ddof = 1).

    """
    return numpy.nanvar(d, ddof = 1)



def nanstd(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The standard deviation of the non-missing data passed; calculated as numpy.nanstd(d, ddof = 1).

    """
    return numpy.nanstd(d, ddof = 1)



def nansem(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The standard error of the non-missing data passed; calculated as scipy.stats.sem(d, nan_policy= 'omit').

    """
    return scipy.stats.sem(d, nan_policy= 'omit')



def value_range(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The range of the data passed; calculated as numpy.nanmax(d) - numpy.nanmin(d).

    """
    min_val = numpy.nanmin(d)
    max_val = numpy.nanmax(d)



    return float(max_val - min_val)



def kurtosis(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The kurtosis of the distribution of the data passed using Pearson's definition; calculated as scipy.stats.kurtosis(d, fisher = False, nan_policy = 'omit').'

    """
    return float(scipy.stats.kurtosis(d, fisher = False, nan_policy = 'omit'))



def skew(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The skew of the distribution of the data passed; calculated as scipy.stats.skew(d, nan_policy = 'omit').

    """
    return float(scipy.stats.skew(d, nan_policy = 'omit'))



def confidence_interval(d, alpha = 0.95, n = None, loc = None, scale = None, decimals = 4):
    """

    Parameters
    ----------
    d : array_like
        The data being passed to the function.

    alpha : decimal (float), optional
        Confidence interval range to be calculated. The default is 0.95.

    n : numeric, optional
        The number of observations - 1. The default is None and is calculated as numpy.count_nonzero(~numpy.isnan(d)) - 1.

    loc : float, optional
        The central measure of tendency to be used. The default is None, which will be calculated as numpy.nanmean(d) (the mean).

    scale : float, optional
        The variability measure to be used. The default is None, which will be calculated as scipy.stats.sem(d, nan_policy= 'omit') (the standard error)

    decimals : integer, optional
        How many decimals places to round to. The default is 4.

    Returns
    -------
    ci_intervals : List
        Returns the confidence interval in a last as the [lower_bound, upper_bound].

    """

    if n == None: n = count(d) - 1
    if loc == None: central = numpy.nanmean(d)
    if scale == None: scaler = nansem(d)

    ci_intervals = list(scipy.stats.t.interval(alpha,
                                               n,
                                               loc = central,
                                               scale = scaler))


    idx = 0
    for value in ci_intervals:
        ci_intervals[idx] = round(value, decimals)
        idx += 1

    return ci_intervals





def l_ci(d, alpha = 0.95, n = None, loc = None, scale = None, decimals = 4):
    """

    Parameters
    ----------
    d : array_like
        The data being passed to the function.

    alpha : decimal (float), optional
        Confidence interval range to be calculated. The default is 0.95.

    n : numeric, optional
        The number of observations - 1. The default is None and is calculated as numpy.count_nonzero(~numpy.isnan(d)) - 1.

    loc : float, optional
        The central measure of tendency to be used. The default is None, which will be calculated as numpy.nanmean(d) (the mean).

    scale : float, optional
        The variability measure to be used. The default is None, which will be calculated as scipy.stats.sem(d, nan_policy= 'omit') (the standard error)

    decimals : integer, optional
        How many decimals places to round to. The default is 4.

    Returns
    -------
    ci_intervals : List
        Returns the lower boud of confidence interval.

    """

    if n == None: n = count(d) - 1
    if loc == None: central = numpy.nanmean(d)
    if scale == None: scaler = nansem(d)

    l_ci, _ = scipy.stats.t.interval(alpha,
                                     n - 1,
                                     loc = central,
                                     scale= scaler)
    return round(l_ci, decimals)



def u_ci(d, alpha = 0.95, n = None, loc = None, scale = None, decimals = 4):
    """

    Parameters
    ----------
    d : array_like
        The data being passed to the function.

    alpha : decimal (float), optional
        Confidence interval range to be calculated. The default is 0.95.

    n : numeric, optional
        The number of observations - 1. The default is None and is calculated as numpy.count_nonzero(~numpy.isnan(d)) - 1.

    loc : float, optional
        The central measure of tendency to be used. The default is None, which will be calculated as numpy.nanmean(d) (the mean).

    scale : float, optional
        The variability measure to be used. The default is None, which will be calculated as scipy.stats.sem(d, nan_policy= 'omit') (the standard error)

    decimals : integer, optional
        How many decimals places to round to. The default is 4.

    Returns
    -------
    ci_intervals : List
        Returns the upper boud of confidence interval.

    """

    if n == None: n = count(d) - 1
    if loc == None: central = numpy.nanmean(d)
    if scale == None: scaler = nansem(d)

    _, u_ci = scipy.stats.t.interval(alpha,
                                     n - 1,
                                     loc= central,
                                     scale= scaler)
    return round(u_ci, decimals)
