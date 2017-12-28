""" Analysis tools for spin entropy paper """

import os
import numpy as np
import pandas as pd
import h5py
from scipy.optimize import curve_fit
import lmfit
from lmfit import Model, Parameters, minimize, fit_report

###################
### HDF IMPORTS ###
###################

def open_hdf5(dat, path=''):
    fullpath = os.path.join(path, 'dat{0:d}.h5'.format(dat))
    return h5py.File(fullpath, 'r')

#########################
### DATA MANIPULATION ###
#########################

def dfdx(f, x, axis = None):
    # returns df(x)/dx
    dx = (x - np.roll(x,1))[1:].mean()
    return np.gradient(f,dx, axis = axis)

def moving_avg(x, y, avgs, axis = None) :
        
    xx = np.cumsum(x, dtype=np.float)
    xx[avgs:] = xx[avgs:] - xx[:-avgs]
    xx = xx[avgs - 1:] / avgs

    if axis==0:
        ret = np.cumsum(y, axis=0, dtype=np.float)
        ret[avgs:] = ret[avgs:] - ret[:-avgs]
        return xx, ret[avgs - 1:] / avgs
    elif axis==1:
        ret = np.cumsum(y, axis=1, dtype=np.float)
        ret[:,avgs:] = ret[:,avgs:] - ret[:,:-avgs]
        return xx, ret[:,avgs - 1:] / avgs
    else:
        ret = np.cumsum(y, dtype=np.float)
        ret[avgs:] = ret[avgs:] - ret[:-avgs]
        return xx, ret[avgs - 1:] / avgs
        
def get_subset(data, bounds):
    """ select cuts of data based on x,y limits
        bounds can be None, which defaults to the extents of x,y """
    
    if(len(bounds)!=2*len(data[2].shape)):
        raise ValueError('Dimensions of bounds and w.extent must match')
    
    extent = [data[0][0], data[0][-1], 
                data[1][0], data[1][-1]]
    
    bs = [b if b else extent[i] for i,b in enumerate(bounds)]

    if(len(data[2].shape)==2):
        ix0 = np.nanargmin(np.abs(data[0]-bs[0]))
        ix1 = np.nanargmin(np.abs(data[0]-bs[1]))
        iy0 = np.nanargmin(np.abs(data[1]-bs[2]))
        iy1 = np.nanargmin(np.abs(data[1]-bs[3]))
        
        return data[0][ix0:ix1], data[1][iy0:iy1], data[2][iy0:iy1,ix0:ix1]
    else:
        raise NotImplemented('1d waves not implemented. Go fix it.')
        
def xy_to_meshgrid(x,y):
    """ returns a meshgrid that makes sense for pcolorgrid
        given z data that should be centered at (x,y) pairs """
    nx = len(x)
    ny = len(y)

    dx = (x[-1] - x[0]) / float(nx - 1)
    dy = (y[-1] - y[0]) / float(ny - 1)

    # shift x and y back by half a step
    x = x-dx/2.0
    y = y-dy/2.0
    
    xn = x[-1]+dx
    yn = y[-1]+dy
    
    return np.meshgrid(np.append(x,xn), np.append(y,yn))
        
###################
### LINE SHAPES ###
###################

def line(x, a, b):
    return a*x + b

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def quadratic(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def i_sense(x, x0, beta, i0, i1, i2):
    """ fit to sensor current """
    arg = (x-x0)/beta
    return -i0*np.tanh(arg) + i1*(x-x0) + i2

def di_sense_simple(x, x0, beta, di0, di2, delta):

    arg = (x-x0)/beta
    return -(0.5)*di0*(arg+delta)*(np.cosh(arg)**-2) + di2

#############
### LINES ###
#############

def dist_2_line(x, y, point, delta):
    # line defined by x, y
    # test if point = [x0,y0] is within delta of line
    test_line = np.stack((x, y)).transpose()
    dist = np.linalg.norm(test_line-point, axis=1)
    return np.any(dist<delta)

def x_intersection(fit0, fit1):
    # fit = (m,b)
    x_int = (fit0[1]-fit1[1])/(fit1[0]-fit0[0])
    return x_int

def y_intersection(fit0, fit1):
    # fit = (m,b)
    x_int = (fit0[1]-fit1[1])/(fit1[0]-fit0[0])
    return fit0[0]*x_int+fit0[1]

####################
### FIT MULTIPLE ###
####################

def i_sense_fit_simultaneous(x, z, centers, widths, x0bounds, constrain = None, span = None):
    """ fit multiple sensor current data simultaneously
        with the option to force one or more parameters to the same value across all 
        datasets """
        
    def i_sense_dataset(params, i, xx):
        # x0, beta, i0, i1, i2
        
        x0 = params['x0_{0:d}'.format(i)]
        beta = params['beta_{0:d}'.format(i)]
        i0 = params['i0_{0:d}'.format(i)]
        i1 = params['i1_{0:d}'.format(i)]
        i2 = params['i2_{0:d}'.format(i)]
        
        return i_sense(xx, x0, beta, i0, i1, i2)
    
    def i_sense_objective(params, xx, zz, idx0, idx1):
        """ calculate total residual for fits to several data sets held
            in a 2-D array"""
        
        n,m = zz.shape
        resid = []
        # make residual per data set
        for i in range(n):
            resid.append(zz[i,idx0[i]:idx1[i]] - i_sense_dataset(params, i, xx[i,idx0[i]:idx1[i]]))
        # now flatten this to a 1D array, as minimize() needs
        return np.concatenate(resid)
    
    # get the dimensions of z
    if(z.ndim==1):
        m = len(z)
        n = 1
        z.shape = (n,m)
    elif(z.ndim==2):
        n,m = z.shape
    else:
        raise ValueError('the shape of zarray is wrong')
    
    # deal with the shape of x
    # should have a number of rows = 1 or number of rows = len(z)
    
    if(x.ndim==1 or x.shape[0]==1):
        x = np.tile(x, (n,1))
    elif(x.shape[0]==n):
        pass
    else:
        raise ValueError('the shape of xarray is wrong')
        
    if(span):
        icenters = np.nanargmin(np.abs(x.transpose()-centers), axis=0)
        ilow = np.nanargmin(np.abs(x.transpose()-(centers-span)), axis=0)
        ihigh = np.nanargmin(np.abs(x.transpose()-(centers+span)), axis=0)
    else:
        ilow = np.zeros(n, dtype=np.int)
        ihigh = -1*np.ones(n, dtype=np.int)
    
    columns = ['x0', 'beta', 'i0', 'i1', 'i2']
    df = pd.DataFrame(columns=columns)
    
    # add constraints specified in the 'constrain' list
    if(constrain):
        
        # create parameters, one per data set
        fit_params = Parameters()

        for i in range(n):
            fit_params.add('x0_{0:d}'.format(i), value=centers[i], min=x0bounds[0], max=x0bounds[1])
            fit_params.add('beta_{0:d}'.format(i), value=widths[i], min=0.2, max=10.0)
            fit_params.add('i0_{0:d}'.format(i), 
                            value=abs(z[i,ilow[i]:ihigh[i]].max()-z[i,ilow[i]:ihigh[i]].min()), min=0.001, max=10.0)
            fit_params.add('i1_{0:d}'.format(i), value=0.1, min=0.0, max=10.0)
            fit_params.add('i2_{0:d}'.format(i), value=z[i,ilow[i]:ihigh[i]].mean(), min=0.0, max=20.0)

        for p in constrain:
            for i in range(1,n):
                fit_params['{0}_{1:d}'.format(p,i)].expr = '{0}_{1:d}'.format(p,0)

        # run the global fit to all the data sets
        m = minimize(i_sense_objective, fit_params, args=(x, z, ilow, ihigh))
    
        valdict = m.params.valuesdict()
        for i in range(n):
            df.loc[i] = [valdict['{0}_{1:d}'.format(c, i)] for c in columns]
    else:
        # no parameters need to be fixed between data sets
        # fit them all separately (much faster)
        for i in range(n):
            p0 = [centers[i], widths[i], abs(z[i,ilow[i]:ihigh[i]].max()-z[i,ilow[i]:ihigh[i]].min()),
                      0.1, z[i,ilow[i]:ihigh[i]].mean()]
            bounds = [(x0bounds[0], 0.2, 0.001, 0.0, 0.0), (x0bounds[1], 10.0, 10.0, 10.0, 20.0)]
            df.loc[i], _ = curve_fit(i_sense, x[i,ilow[i]:ihigh[i]], z[i,ilow[i]:ihigh[i]], p0=p0, bounds=bounds)
                          
    return df

def di_fit_simultaneous(x, z, centers, widths, x0bounds, constrain = None, fix = None, span = None):
    
    def di_dataset(params, i, xx):
        """ Weak localization peak fitting function. Adapted from Igor code. """
        
        x0 = params['x0_{0:d}'.format(i)]
        beta = params['beta_{0:d}'.format(i)]
        di0 = params['di0_{0:d}'.format(i)]
        di2 = params['di2_{0:d}'.format(i)]
        delta = params['delta_{0:d}'.format(i)]
        
        return di_sense_simple(xx, x0, beta, di0, di2, delta)
    
    def di_objective(params, xx, zz, idx0, idx1):
        """ calculate total residual for fits to several data sets held
            in a 2-D array, and modeled by Gaussian functions"""
        
        n,m = zz.shape
        resid = []
        # make residual per data set
        for i in range(n):
            resid.append(zz[i,idx0[i]:idx1[i]] - di_dataset(params, i, x[i,idx0[i]:idx1[i]]))
        # now flatten this to a 1D array, as minimize() needs
        return np.concatenate(resid)
    
    # get the dimensions of z
    if(z.ndim==1):
        m = len(z)
        n = 1
        z.shape = (n,m)
    elif(z.ndim==2):
        n,m = z.shape
    else:
        raise ValueError('the shape of zarray is wrong')
    
    # deal with the shape of x
    # should have a number of rows = 1 or number of rows = len(z)
    
    if(x.ndim==1 or x.shape[0]==1):
        x = np.tile(x, (n,1))
    elif(x.shape[0]==n):
        pass
    else:
        raise ValueError('the shape of xarray is wrong')
        
    if(span):
        icenters = np.nanargmin(np.abs(x.transpose()-centers), axis=0)
        ilow = np.nanargmin(np.abs(x.transpose()-(centers-span)), axis=0)
        ihigh = np.nanargmin(np.abs(x.transpose()-(centers+span)), axis=0)
    else:
        ilow = np.zeros(n, dtype=np.int)
        ihigh = -1*np.ones(n, dtype=np.int)
    
    columns = ['x0', 'beta', 'di0', 'di2', 'delta']
    df = pd.DataFrame(columns=columns)
    
    # add constraints specified in the 'constrain' list
    if(constrain or fix):
        
        # create parameters, one per data set
        fit_params = Parameters()

        for i in range(n):
            fit_params.add('x0_{0:d}'.format(i), value=centers[i], min=x0bounds[0], max=x0bounds[1])
            fit_params.add('beta_{0:d}'.format(i), value=widths[i], min=0.2, max=10.0)
            fit_params.add('di0_{0:d}'.format(i), 
                               value=max(abs(z[i,ilow[i]:ihigh[i]].min()), abs(z[i,ilow[i]:ihigh[i]].max())), min=0.0, max=0.5)
            fit_params.add('di2_{0:d}'.format(i), value=(z[i,ilow[i]]+z[i,ihigh[i]])/2.0, min=-0.01, max=0.01)
            fit_params.add('delta_{0:d}'.format(i), value=0.0, min=-2.0, max=2.0)

        if(constrain):
            for p in constrain:
                for i in range(1,n):
                    fit_params['{0}_{1:d}'.format(p,i)].expr = '{0}_{1:d}'.format(p,0)
                    
        if(fix):
            for p in fix:
                for i in range(n):
                    fit_params['{0}_{1:d}'.format(p,i)].vary = False
                    
        # run the global fit to all the data sets
        m = minimize(di_objective, fit_params, args=(x, z, ilow, ihigh))
    
        valdict = m.params.valuesdict()
        for i in range(n):
            df.loc[i] = [valdict['{0}_{1:d}'.format(c, i)] for c in columns]
    else:
        # no parameters need to be fixed between data sets
        # fit them all separately (much faster)
        for i in range(n):
            p0 = [centers[i], widths[i], max(abs(z[i,ilow[i]:ihigh[i]].min()), abs(z[i,ilow[i]:ihigh[i]].max())),
                  (z[i,ilow[i]]+z[i,ihigh[i]])/2.0, 0.0]
            bounds = [(x0bounds[0], 0.2, 0.0, -0.05, -2.0), (x0bounds[1], 10.0, 0.5, 0.05, 2.0)]
            df.loc[i], _ = curve_fit(di_sense_simple, x[i,ilow[i]:ihigh[i]], z[i,ilow[i]:ihigh[i]], p0=p0, bounds=bounds)
                          
    return df