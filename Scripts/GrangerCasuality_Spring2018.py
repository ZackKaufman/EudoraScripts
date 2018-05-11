# import needed packages
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dt, num2date
import xarray as xr
from statsmodels.tsa.stattools import grangercausalitytests as granger
from statsmodels.tsa.seasonal import seasonal_decompose as deseasonalize
import statsmodels.tsa.api as sm
import scipy

############################################################
# import variable data (SH lats only) from /LENSoutput directory
# data begins December of year 1 for easier data management
# time dimension may be restricted to small size
# to speed up analysis at first

ICEFRAC_file = dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.ICEFRAC.04020101-04991231.nc')
ICEFRAC = ICEFRAC_file.variables['ICEFRAC'][333:3983,0:96,:]
# convert ICEFRAC from decimal to %
ICEFRAC = ICEFRAC * 100

time = ICEFRAC_file.variables['time'][333:3983]
# set day 1 to 0 instead of "days since year 256 (see header file)"
time = time - 42674
# get total years in dataset from time (to be used later)
yearcount = len(time)/365

FSNS_file = dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FSNS.04020101-04991231.nc')
FSNS = FSNS_file.variables['FSNS'][333:3983,0:96,:]
###########################################################

# plot three years worth of data at random grid point
# and check for seasonal trend
# must export figure to local machine to view, plt.show() invalid on ssh server

plt.plot(ICEFRAC[:,27,100], 'b')
plt.plot(FSNS[:,27,100], 'r')
os.chdir('../draftfigures')
plt.savefig('10yr_ICE_FSNS_may7.png')
plt.close()
###########################################################

# VAR(p) model for Granger Causality most effective when analyzing one season
# function: isolate season of interest (DJF in this case)
def seasonselect(var,newtimelength):
    x = np.vsplit(var, ((len(var)/365)))
    x= np.array(x)
    x = x[:,0:90,:,:]
    x = np.reshape(x,(newtimelength,96,288))
    return x

# isolate DJF for all variables
newtimelength = yearcount*90
ICEFRAC_DJF = seasonselect(ICEFRAC,int(newtimelength))
FSNS_DJF = seasonselect(FSNS,int(newtimelength))

# plot data
plt.plot(ICEFRAC_DJF[:,27,100], 'b')
plt.plot(FSNS_DJF[:,27,100], 'r')
plt.savefig('10yr_ICE_FSNS_DJF_may7.png')
plt.close()

###########################################################
# Data Management: Remove seasonal trends to achieve stationarity

#Function: Apply a difference transform to remove the mean annual harmonic.
# The output of this function are the residuals of the mean periodic trend,
# assuming that there is no linear trend present. Year 1 is sacrificed in the
# process.
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

ICEFRAC_detrend_DJF = difference(ICEFRAC_DJF, 90)
ICEFRAC_detrend_DJF = np.array(ICEFRAC_detrend_DJF)

FSNS_detrend_DJF = difference(FSNS_DJF, 90)
FSNS_detrend_DJF = np.array(FSNS_detrend_DJF)

#Plot detrended data
plt.plot(ICEFRAC_detrend_DJF[:,27,100], 'b')
plt.plot(FSNS_detrend_DJF[:,27,100], 'r')
plt.savefig('9yrstddetrend_ICE_FSNS_may7.png')
plt.close()

###########################################################

# smooth data in 25-day chunks (review this later)
ICEFRAC_smooth_detrend_DJF = scipy.signal.savgol_filter(ICEFRAC_detrend_DJF, \
window_length=5,polyorder=2,axis=0)
FSNS_smooth_detrend_DJF = scipy.signal.savgol_filter(FSNS_detrend_DJF,\
window_length=5,polyorder=2,axis=0)


# plot data
plt.plot(ICEFRAC_smooth_detrend_DJF[:,27,100], 'b')
plt.plot(FSNS_smooth_detrend_DJF[:,27,100], 'r')
plt.savefig('9yrstddetrendsmooth_ICE_FSNS_may7.png')
plt.close()

###########################################################
# normalize data into anomalies
# function: subtract functions mean and divide by standard deviation:
def normalize(data):
    data = ((data - np.mean(data))/np.std(data))
    return data

ICEFRAC_norm_smooth_detrend_DJF = normalize(ICEFRAC_smooth_detrend_DJF)
FSNS_norm_smooth_detrend_DJF = normalize(FSNS_smooth_detrend_DJF)

plt.plot(ICEFRAC_norm_smooth_detrend_DJF[:,27,100], 'b')
plt.plot(FSNS_norm_smooth_detrend_DJF[:,27,100], 'r')
plt.savefig('9yr_normdetrendDJF_may7.png')
plt.close()
###########################################################
# conduct granger causality test

# function: create VARmodel, input is 2d array of shape (n_obs,n_var)

def VARmodel(dataset):
    VARmodel = sm.VAR(dataset)
    VARmodel_fit = VARmodel.fit(ic='aic',trend='c')
    return VARmodel_fit

# function: conduct causality test, returns binary depending on
# rejection or failed rejection of null-hyp

def grangertest(model,predictand,predictor):
    test = model.test_causality(str(predictand),str(predictor),verbose=False)
    x=0
    if 'reject' in test.values():
        x = x+1
    else:
        x = x+2
    return x

ICEFRAC_in = ICEFRAC_norm_smooth_detrend_DJF
FSNS_in = FSNS_norm_smooth_detrend_DJF

grangergrid = ICEFRAC_in*0
grangergrid = np.mean(grangergrid,axis=0)

for i,j in np.ndindex(ICEFRAC_in.shape[1:]) and np.ndindex(FSNS_in.shape[1:]):
    y1 = ICEFRAC_in[:,i,j]
    y2 = FSNS_in[:,i,j]
    # the followinng process adds small amounts of random noise
    # this eliminates the singular matrix problem
    dataset = [y1,y2]+.00000001*np.random.rand(2,810)
    dataset = np.array(dataset)
    dataset = dataset.T
    ICEFRAC_FSNS_VARmodel = VARmodel(dataset)
    result = grangertest(ICEFRAC_FSNS_VARmodel,'y1','y2')
    grangergrid[i,j] = result[i,j]
    print i,j,grangergrid.shape
    if i > 1:
        break
    finalgrid = grangergrid
    return finalgrid
    print grangergrid.shape
    print grangergrid[27,100]
    print grangergrid[:,105]
    print grangergrid[:,155]





i = 0
j = 0
for i,j in zip(np.ndindex(ICEFRAC_in.shape[1:]),np.ndindex(FSNS_in.shape[1:])):
    dataset = [ICEFRAC_in[:,i,j],FSNS_in[:,i,j]]
    dataset = np.array(dataset)
    print dataset.shape
    y1 = ICEFRAC_in[:,i,j]
    y2 = FSNS_in[:,i,j]
    print y1.shape
    dataset = [y1,y2]
    dataset = np.array(dataset)
    dataset = dataset.T
    print dataset.shape


    for k,l in np.ndindex(FSNS_in.shape[1:]):
        y2 = FSNS_in.shape[:,k,l]
        print y2
        dataset = [y1,y2]
        dataset = np.array(dataset)
        dataset = dataset.T
        ICEFRAC_FSNS_VARmodel = VARmodel(dataset)
        result = grangertest(ICEFRAC_FSNS_VARmodel,'y1','y2')




    x = a[:,i,j]
    print x.shape



grangerICE = ICEFRAC_norm_smooth_detrend_DJF[:,27,100]
grangerFSNS = FSNS_norm_smooth_detrend_DJF[:,27,100]
dataset = [grangerICE,grangerFSNS]
dataset = np.array(dataset)
dataset = dataset.T

ICEFRAC_FSNS_VARmodel = VARmodel(dataset)
result = grangertest(ICEFRAC_FSNS_VARmodel,'y1','y2')
print result
