""" Analysis tools for spin entropy paper """

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
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

################
### PLOTTING ###
################

def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    # credit:  https://gist.github.com/phobson/7916777
    
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero
    
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). 
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
      
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
        
    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def add_subplot_id(ax, id_lttr, loc, fontsize=16):
    ax.text(*loc, r'{0:s}'.format(id_lttr), transform=ax.transAxes, 
            fontsize=fontsize, fontweight='bold')

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

MU_B = 5.7883818012e-5 # eV/T
K_B = 8.6173303e-5 # eV/K

def line(x, a, b):
    return a*x + b

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def i_sense(x, x0, theta, i0, i1, i2):
    """ fit to sensor current """
    arg = (x-x0)/(2*theta)
    return -i0*np.tanh(arg) + i1*(x-x0) + i2

def di_sense_simple(x, x0, theta, di0, di2, epsilon):
    """ fit charge sensor lock in signal """
    arg = (x-x0)/(2*theta)
    return -1.0*di0*(arg+0.5*epsilon)*(np.cosh(arg)**-2) + di2

def p_up(field, temp, g, de):
    return 1/(1+np.exp(-(g*MU_B*field-de)/(K_B*temp)))

def p_down(field, temp, g, de):
    return 1/(1+np.exp(+(g*MU_B*field-de)/(K_B*temp)))

def gibbs_entropy(field, a, b, temp, g, de):
    return a*(-p_up(field, temp, g, de)*np.log(p_up(field, temp, g, de)) \
            -p_down(field, temp, g, de)*np.log(p_down(field, temp, g, de))) + b

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
        # x0, theta, i0, i1, i2
        
        x0 = params['x0_{0:d}'.format(i)]
        theta = params['theta_{0:d}'.format(i)]
        i0 = params['i0_{0:d}'.format(i)]
        i1 = params['i1_{0:d}'.format(i)]
        i2 = params['i2_{0:d}'.format(i)]
        
        return i_sense(xx, x0, theta, i0, i1, i2)
    
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
    
    columns = ['x0', 'theta', 'i0', 'i1', 'i2']
    df = pd.DataFrame(columns=columns)
    
    # add constraints specified in the 'constrain' list
    if(constrain):
        
        # create parameters, one per data set
        fit_params = Parameters()

        for i in range(n):
            fit_params.add('x0_{0:d}'.format(i), value=centers[i], min=x0bounds[0], max=x0bounds[1])
            fit_params.add('theta_{0:d}'.format(i), value=widths[i], min=0.05, max=10.0)
            fit_params.add('i0_{0:d}'.format(i), 
                            value=abs(z[i,ilow[i]:ihigh[i]].max()-z[i,ilow[i]:ihigh[i]].min()),
                            min=0.001, max=10.0)
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
            bounds = [(x0bounds[0], 0.05, 0.001, 0.0, 0.0), (x0bounds[1], 10.0, 10.0, 10.0, 20.0)]
            df.loc[i], _ = curve_fit(i_sense, x[i,ilow[i]:ihigh[i]], z[i,ilow[i]:ihigh[i]], p0=p0, bounds=bounds)
                          
    return df

def di_fit_simultaneous(x, z, centers, widths, x0bounds, 
                        constrain = None, fix = None, span = None, 
                        nboot=None, bootstat='bounds'):
    
    def di_bootstrap_eps(mboot, xx, zz, fit_params, jlow, jhigh, pp0, bbounds):
        """ bootstrap estimate of errors on epsilon for single curve fit 
            Following this: http://www.phas.ubc.ca/~oser/p509/Lec_20.pdf """
        
        # create zfit and resid, both of which have shape=z.shape
        zfit = di_sense_simple(xx, *fit_params)
        resid = zz - zfit
        
        boot_results = np.zeros((mboot,len(fit_params)))
        for k in range(mboot):
            ztest = zfit + np.random.choice(resid.flatten(), size=zfit.shape)
            out, _ = curve_fit(di_sense_simple, xx[jlow:jhigh], 
                                ztest[jlow:jhigh], p0=pp0, bounds=bbounds)
            boot_results[k] = out
        
        if bootstat=='bounds':
            return np.percentile(boot_results[:,-1], [2.5, 97.5])
        elif bootstat=='std':
            return np.array([boot_results[:,-1].std(), boot_results[:,-1].std()])
    
    def di_dataset(params, i, xx):
        
        x0 = params['x0_{0:d}'.format(i)]
        theta = params['theta_{0:d}'.format(i)]
        di0 = params['di0_{0:d}'.format(i)]
        di2 = params['di2_{0:d}'.format(i)]
        epsilon = params['epsilon_{0:d}'.format(i)]
        
        return di_sense_simple(xx, x0, theta, di0, di2, epsilon)
    
    def di_objective(params, xx, zz, idx0, idx1):
        """ calculate total residual for fits to several data sets held
            in a 2-D array """
        
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
    
    columns = ['x0', 'theta', 'di0', 'di2', 'epsilon']
    df = pd.DataFrame(columns=columns)
    
    # add constraints specified in the 'constrain' list
    if(constrain or fix):
        
        # create parameters, one per data set
        fit_params = Parameters()

        for i in range(n):
            fit_params.add('x0_{0:d}'.format(i), value=centers[i], min=x0bounds[0], max=x0bounds[1])
            fit_params.add('theta_{0:d}'.format(i), value=widths[i], min=0.05, max=10.0)
            fit_params.add('di0_{0:d}'.format(i), 
                            value=0.5*max(abs(z[i,ilow[i]:ihigh[i]].min()),
                                          abs(z[i,ilow[i]:ihigh[i]].max())), 
                           min=0.0, max=0.5)
            fit_params.add('di2_{0:d}'.format(i), value=(z[i,ilow[i]]+z[i,ihigh[i]])/2.0, 
                           min=-0.01, max=0.01)
            fit_params.add('epsilon_{0:d}'.format(i), value=0.0, min=-2.0, max=2.0)

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
        if nboot:
            eps_err = np.zeros((n,2))
        for i in range(n):
            p0 = [centers[i], widths[i], 
                  max(abs(z[i,ilow[i]:ihigh[i]].min()), abs(z[i,ilow[i]:ihigh[i]].max())),
                  (z[i,ilow[i]]+z[i,ihigh[i]])/2.0, 0.0]
            bounds = [(x0bounds[0], 0.05, 0.0, -0.05, -2.0), (x0bounds[1], 10.0, 0.5, 0.05, 2.0)]
            df.loc[i], _ = curve_fit(di_sense_simple, x[i,ilow[i]:ihigh[i]], 
                                             z[i,ilow[i]:ihigh[i]], p0=p0, bounds=bounds)
            
            if nboot:
                eps_err[i] = di_bootstrap_eps(nboot, x[i], z[i], df.loc[i],
                                     ilow[i], ihigh[i], p0, bounds)
        if nboot:
            return df, eps_err
            
    return df