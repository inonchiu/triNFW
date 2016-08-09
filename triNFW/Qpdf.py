#!/usr/bin/env python

######################################
#
# This is the simple module to quantify the given sample typically drawn from a distribution.
#
######################################

import numpy as np
from math import *
# do we have astropy?
try:
    import astropy.stats as astrostats
    have_astropy    =   True
except ImportError:
    have_astropy    =   False


def quan_sample(data, biweight_c = 9.0, verbose = True, plotme = False):
    """
    This function calculates the properties that are used to quantify the given sample.

    Parameters:
        -`data`: 1d array. This is the sample of data.
        -`biweight_c`: float. The input argument for biweight estimator.
        -`verbose`: bool. True for outputing the results on standard output.
        -`plotme`: bool. True for creating the plot - matplotlib required.

    Returns:
        -``:
    """
    # sanitize - using masked array.
    data                =       np.ma.array(data, mask = ~np.isfinite(data))
    biweight_c          =       float(biweight_c)

    # hist
    hist, edges         =       np.histogram(data.compressed(), bins = 100)
    bins                =       0.5 * (edges[1:] + edges[:-1])

    # deriving properties
    return_mean         =       np.ma.mean( data )
    return_median       =       np.ma.median( data )
    return_std          =       np.ma.std( data )
    return_max          =       np.ma.max( data )
    return_min          =       np.ma.min( data )
    if  have_astropy:
        return_bi_loc   =       astrostats.biweight_location(data, c = biweight_c)
        return_bi_scl   =       astrostats.biweight_midvariance(data, c = biweight_c)
        return_mad      =       astrostats.median_absolute_deviation(data)
        return_mad_std  =       astrostats.mad_std(data)
    else:
        return_bi_loc   =       np.nan
        return_bi_scl   =       np.nan
        return_mad      =       np.nan
        return_mad_std  =       np.nan

    # verbose?
    if   verbose:
        print "#", "min:",                 return_min
        print "#", "max:",                 return_max
        print "#", "mean:",                return_mean
        print "#", "median:",              return_median
        print "#", "std:",                 return_std
        print "#", "biweight_location:",   return_bi_loc
        print "#", "biweight_scale:",      return_bi_scl
        print "#", "mad:",                 return_mad
        print "#", "mad_std:",             return_mad_std

    # plotme?
    if  plotme:
        import matplotlib.pyplot as pyplt
        pyplt.plot(bins, hist, "k-")
        pyplt.show()

    # return
    return {"max":    return_max,
            "min":    return_min,
            "mean":   return_mean,
            "median": return_median,
            "std":    return_std,
            "bi_loc": return_bi_loc,
            "bi_scl": return_bi_scl,
            "mad":    return_mad,
            "mad_std":return_mad_std,}

# testing
if    __name__ == "__main__":
    nsample     =       1000
    sample      =   np.random.normal(loc = 0.0, scale = 1.0, size = nsample) + \
                    10.0**np.random.normal(loc = 0.0, scale = 0.6, size = nsample)
    # tweaking
    sample[[2,3,4,5,6]] = np.nan
    sample      =   np.ma.array(sample, mask = ~np.isfinite(sample))

    # how is it?
    quan_sample(sample)
