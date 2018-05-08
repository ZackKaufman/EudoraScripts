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

#confirm stationarity with summary statistics and histograms.
# Do mean and variance change over time?
X = ICEFRAC_detrend_DJF[:,27,100]
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
plt.hist(X)
plt.savefig('ICEhist_may7.png')
plt.close()

X = FSNS_detrend_DJF[:,27,100]
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
plt.hist(X)
plt.savefig('FSNShist_may7.png')
plt.close()

###########################################################

# function: smooth data to remove high-frequency noise, only works for 1d data for now
def smooth(y,window_len=11,window='hanning'):
    for x in np.nditer(y, flags=['external_loop']):
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat':
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')
            x=np.convolve(w/w.sum(),s,mode='valid')
        return y

# smooth data in 5-day chunks
ICEFRAC_smooth_detrend_DJF = scipy.signal.savgol_filter(ICEFRAC_detrend_DJF,window_length=25,polyorder=2,axis=0)
FSNS_smooth_detrend_DJF = scipy.signal.savgol_filter(FSNS_detrend_DJF,window_length=25,polyorder=2,axis=0)


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

ICEFRAC_norm_detrend_DJF = normalize(ICEFRAC_detrend_DJF)
FSNS_norm_detrend_DJF = normalize(FSNS_detrend_DJF)

plt.plot(ICEFRAC_norm_detrend_DJF[:,27,100], 'b')
plt.plot(FSNS_norm_detrend_DJF[:,27,100], 'r')
plt.savefig('9yr_normdetrendDJF_may7.png')
plt.close()
