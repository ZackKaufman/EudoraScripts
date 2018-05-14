# import needed packages
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as c
from netCDF4 import Dataset as dt, num2date
from mpl_toolkits.basemap import Basemap, cm, shiftgrid
import statsmodels.tsa.api as sm
import scipy


############################################################
# import variable data (SH lats only) from /LENSoutput directory
# data begins December of year 1 for easier data management
# time dimension may be restricted to 10-years
# to speed up analysis processes at first
ICEFRAC_file = dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.ICEFRAC.04020101-04991231.nc')
FSNS_file = dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FSNS.04020101-04991231.nc')
ICEFRAC = ICEFRAC_file.variables['ICEFRAC'][333:3983,0:60,:]
FSNS = FSNS_file.variables['FSNS'][333:3983,0:60,:]

# convert ICEFRAC from decimal to %
ICEFRAC = ICEFRAC * 100

# import time data  to aid in seasonal selection
time = ICEFRAC_file.variables['time'][333:3983]
# set day 1 to 0 instead of "days since year 256 (see header file)"
time = time - 42674
# import LAT and LON data from one variable for map generation
lons = ICEFRAC_file.variables['lon'][:]
lats = ICEFRAC_file.variables['lat'][0:60]
###########################################################
###########################################################

# VAR(p) model for Granger Causality most effective when analyzing one season
# function: isolate season of interest (DJF in this case)
def seasonselect(var,newtimelength):
    x = np.vsplit(var, ((len(var)/365)))
    x= np.array(x)
    x = x[:,0:90,:,:]
    x = np.reshape(x,(newtimelength,60,288))
    return x

# isolate DJF for all variables
yearcount = len(time)/365
newtimelength = yearcount*90
ICEFRAC_DJF = seasonselect(ICEFRAC,int(newtimelength))
FSNS_DJF = seasonselect(FSNS,int(newtimelength))

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

###########################################################

def movingaverage_5day(values,window=5):
    weights = np.repeat(1.0,window)/window
    smas = np.convolve(values,weights,mode='valid')
    smas_5day = smas[::5]
    return smas_5day

ICEFRAC_smooth_detrend_DJF = \
np.apply_along_axis(movingaverage_5day,0,ICEFRAC_detrend_DJF)

FSNS_smooth_detrend_DJF = \
np.apply_along_axis(movingaverage_5day,0,FSNS_detrend_DJF)

###########################################################
# normalize data into anomalies
# function: subtract functions mean and divide by standard deviation:

def normalize(data):
    data = ((data - np.mean(data))/np.std(data))
    return data

ICEFRAC_norm_smooth_detrend_DJF = normalize(ICEFRAC_smooth_detrend_DJF)
FSNS_norm_smooth_detrend_DJF = normalize(FSNS_smooth_detrend_DJF)

###########################################################
# conduct granger causality test, main implementation

# function: create VARmodel, input is 2d array of shape (n_obs,n_var)
def VARmodel(dataset):
    VARmodel = sm.VAR(dataset)
    VARmodel_fit = VARmodel.fit(ic='aic',trend='c')
    return VARmodel_fit

# function: conduct causality test for one model, returns binary depending on
# rejection or failed rejection of null-hypothesis
def grangertest(model,predictand,predictor):
    test = model.test_causality(str(predictand),str(predictor),verbose=False)
    x=0
    if 'reject' in test.values():
        x = x+1
    else:
        x = x+2
    return x

# shorten the names of the curated input variables. values remain unchanged.
# ICEFRAC_check is used to exclude data points where ICEFRAC is almost
# non-existent
ICEFRAC_in = ICEFRAC_norm_smooth_detrend_DJF
FSNS_in = FSNS_norm_smooth_detrend_DJF
ICEFRAC_check = np.max(ICEFRAC_DJF,axis=0)

# initialize 2d lat/lon array with dimensions equal to input variable(s) lat/lon
# float values = 0 to start
# values are then assigned by the main implementation with VARmodels
# grangergrid with filled values are then plotted on a basemap
grangergrid = ICEFRAC_in*0
grangergrid = np.mean(grangergrid,axis=0)

# main implementation analyzes time series arrays for each lat/lon grid point
for i,j in np.ndindex(ICEFRAC_in.shape[1:]) and np.ndindex(FSNS_in.shape[1:]):
    y1 = ICEFRAC_in[:,i,j]
    y2 = FSNS_in[:,i,j]
    icecheck = ICEFRAC_check[i,j]
    dataset = (np.array([y1,y2])).T
    # don't fit a VARmodel if ICEFRAC is at or near constant
    #(i.e. approx. 0 throughout time)
    if icecheck < 5:
        result = 0
        grangergrid[i,j] = result
    else:
        modelfit = VARmodel(dataset)
        # if AIC chooses a lag-0 model, avoid grangertest and output the
        #grid point where it occurs to see what went wrong
        if modelfit.df_model==1:
            result = 3
            grangergrid[i,j] = result
            print (i,j),result
        else:
            result = grangertest(modelfit,'y2','y1')
            grangergrid[i,j] = result





###########################################################
# plot desired grids on map
# function, input grid to be plotted, title

def mapplot(data,title,levels=None,fillcontinents=False):
        lons = ICEFRAC_file.variables['lon'][:]
        lats = ICEFRAC_file.variables['lat'][0:60]
        fig = plt.figure(figsize=[12,15])  # a new figure window
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.set_title(str(title), fontsize=14)
        map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=-40, \
        llcrnrlon=-180,urcrnrlon=180,resolution='c', ax=ax)
        map.drawcoastlines()
        if fillcontinents is True:
            map.fillcontinents(color='#ffe2ab')
        # draw parallels and meridians.
        map.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
        map.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        # shift data so lons go from -180 to 180 instead of 0 to 360.
        data,lons = shiftgrid(180.,data,lons,start=False)
        llons, llats = np.meshgrid(lons, lats)
        x,y = map(llons,llats)
        cmap = plt.cm.get_cmap('spectral')
        #bounds=[-1,0,1,2,3]
        cs = map.contourf(x,y,data,cmap=cmap,levels=levels,shading='interp')
        ## make a color bar
        fig.colorbar\
        (cs, ax=ax,cmap=cmap, orientation='horizontal')
        # return lons to their original state
        lons = ICEFRAC_file.variables['lon'][:]
        return fig

# make basemap 1
os.chdir('../draftfigures')
ICEFRAC_DJF = np.mean(ICEFRAC_DJF,axis=0)
figure_ICEFRAC = mapplot(ICEFRAC_DJF,'Mean DJF Ice fraction (%)',\
[.01,.1,1,10,20,30,40,50,60,70,80,90,100])
figure_ICEFRAC.savefig('mean_Ice_fraction_%_may14.png')
plt.close()
# return var to initial state
ICEFRAC_DJF = seasonselect(ICEFRAC,int(newtimelength))

#make basemap 2
ICEFRAC_DJF = np.max(ICEFRAC_DJF,axis=0)
figure_ICEFRAC = mapplot(ICEFRAC_DJF,'Max DJF Ice fraction (%)',\
[.01,.1,1,10,20,30,40,50,60,70,80,90,100])
figure_ICEFRAC.savefig('max_Ice_fraction_%_may14.png')
plt.close()
# return var to initial state
ICEFRAC_DJF = seasonselect(ICEFRAC,int(newtimelength))

#basemap 3
ICEFRAC_detrend_DJF = np.mean(ICEFRAC_detrend_DJF,axis=0)
figure_ICEFRAC = mapplot(ICEFRAC_detrend_DJF,\
'Mean detrended DJF Ice fraction (%)')
figure_ICEFRAC.savefig('meandetrend_Ice_fraction_%_may14.png')
plt.close()
# return var to initial state
ICEFRAC_detrend_DJF = difference(ICEFRAC_DJF, 90)
ICEFRAC_detrend_DJF = np.array(ICEFRAC_detrend_DJF)

#basemap 4
figure_granger = mapplot(grangergrid,\
'Grangerresults_may14#1',levels=[0,1,2,3])
figure_granger.savefig('granger1_may14.png')
plt.close()
# return var to initial state
grangergrid = ICEFRAC_in*0
grangergrid = np.mean(grangergrid,axis=0)




fig = plt.figure(figsize=[12,15])  # a new figure window
ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
ax.set_title('ICEFRAC -> Net incoming SR? (10-yr pi-control)', fontsize=14)

map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=-40,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c', ax=ax)

map.drawcoastlines()
#map.fillcontinents(color='#ffe2ab')
# draw parallels and meridians.
map.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
map.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

# shift data so lons go from -180 to 180 instead of 0 to 360.
grangergrid,lons = shiftgrid(180.,grangergrid,lons,start=False)
llons, llats = np.meshgrid(lons, lats)
x,y = map(llons,llats)
bounds=[-1,0,1,2,3]

cs = map.contourf(x,y,grangergrid, levels=bounds,shading='interp')

## make a color bar
fig.colorbar(cs,boundaries=bounds, ax=ax, orientation='horizontal')
os.chdir('../draftfigures')
fig.savefig()
plt.close()

# remake empty grid and lat/lon for next plot
lons = ICEFRAC_file.variables['lon'][:]
grangergrid = ICEFRAC_in*0
grangergrid = np.mean(grangergrid,axis=0)
