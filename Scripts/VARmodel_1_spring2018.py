# import required packages
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
# import VAR model variables of interest:
# FSNS (net surface shortwave radiation)
# FSNSC (net surface shortwave radiation: clear sky)
# FLNS (net surface longwave radiation)
# FLNSC (not surface longwave radiation: clear sky)
# ICEFRAC (ice fraction)
# TGCLDLWP (total grid cloud liquid water path)
# CLDTOT (vertically integrated cloud fraction)

FSNS_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FSNS.15000101-15991231.nc')
FSNSC_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FSNSC.15000101-15991231.nc')
FLNS_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FLNS.15000101-15991231.nc')
FLNSC_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FLNSC.15000101-15991231.nc')
ICEFRAC_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.ICEFRAC.15000101-15991231.nc')
TGCLDLWP_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h2.TGCLDLWP.1500010100Z-1510123118Z.nc')
CLDTOT_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h2.CLDTOT.1500010100Z-1510123118Z.nc')

###########################################################
# for monthly climatologies, we trim time dimension to start at march 1st

# for inputs starting March 1 year 1501:
#  6hr data (clouds only), march1_startindex = 235
#  daily data, march1_startindex = 423

#  we also trim data to end after 9 annual cycles (limit of cloud data)
# for inputs ending feb 28 year 1510:
# for 6 hr data (clouds only), feb28_endindex = 13375
# for daily data, feb28_endindex = 3708

# We are interested in the Southern Ocean
# Latitude dimension is trimmed to portion of S Hem. (saves computing time)
# Longitude dimension is included in full

ICEFRAC = ICEFRAC_file.variables['ICEFRAC'][423:3708,0:60,:]
FSNS = FSNS_file.variables['FSNS'][423:3708,0:60,:]
FSNSC = FSNSC_file.variables['FSNSC'][423:3708,0:60,:]
FLNS = FLNS_file.variables['FLNS'][423:3708,0:60,:]
FLNSC = FLNSC_file.variables['FLNSC'][423:3708,0:60,:]
TGCLDLWP = TGCLDLWP_file.variables['TGCLDLWP'][235:13375,0:60,:]
CLDTOT = CLDTOT_file.variables['CLDTOT'][235:13375,0:60,:]

# ICEFRAC and CLDTOT units are converted from decimal to percent

ICEFRAC = ICEFRAC * 100
CLDTOT = CLDTOT * 100

# clear sky radiation fluxes are subtracted from the net
# to obtain SW and LW cloud radiative effects

SWCRE = FSNSC - FSNS
LWCRE = FLNSC - FLNS
NETCRE = LWCRE - SWCRE
###########################################################
# We must reduce cloud variables from hourly resolution to daily resolution
# function: average time dimension into non-overlapping chunks

# convolution is the fasted way to achieve this
# use mode = valid to eliminate boundary effects, mode=full to preserve dim's

def movingaverage(values,window,mode):
    weights = np.repeat(1.0,window)/window
    average = np.convolve(values,weights,mode=str(mode))
    average = average[::int(window)]
    return average

# apply moving average function along time axis for cloud data

TGCLDLWP_daily = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=TGCLDLWP,window=4,mode='valid')
TGCLDLWP = TGCLDLWP_daily

CLDTOT_daily = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=CLDTOT,window=4,mode='valid')
CLDTOT = CLDTOT_daily

###########################################################

# We'd like to split our data into seasons, analyzing one at a time.
# function: isolate season of interest
# function assumes lat dimension = 60 and lon dimension = 288
# function assumes time dimension begins march 1st

def seasonselect(var,season):
    yearnumber = len(var)/365
    newtimelength = yearnumber * 90
    x = np.vsplit(var,yearnumber)
    x= np.array(x)
    if str(season) == 'MAM':
        x = x[:,0:90,:,:]
    elif str(season) == 'JJA':
        x = x[:,90:180,:,:]
    elif str(season) == 'SON':
        x = x[:,180:270,:,:]
    elif str(season) == 'DJF':
        x = x[:,270:360,:,:]
    x = np.reshape(x,(newtimelength,60,288))
    return x

# isolate DJF (austral summer) for all variables

ICEFRAC_DJF = seasonselect(ICEFRAC,'DJF')
SWCRE_DJF = seasonselect(SWCRE,'DJF')
LWCRE_DJF = seasonselect(LWCRE,'DJF')
TGCLDLWP_DJF = seasonselect(TGCLDLWP,'DJF')
CLDTOT_DJF = seasonselect(CLDTOT,'DJF')
NETCRE_DJF = seasonselect(NETCRE,'DJF')
# isolate MAM (austral fall) for all variables

ICEFRAC_MAM = seasonselect(ICEFRAC,'MAM')
SWCRE_MAM = seasonselect(SWCRE,'MAM')
LWCRE_MAM = seasonselect(LWCRE,'MAM')
TGCLDLWP_MAM = seasonselect(TGCLDLWP,'MAM')
CLDTOT_MAM = seasonselect(CLDTOT,'MAM')
NETCRE_MAM = seasonselect(NETCRE,'MAM')

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
    diff = np.array(diff)
    return diff

# for seasonally separated data, harmonic = 90 days

ICEFRAC_detrend_DJF = difference(ICEFRAC_DJF, 90)
SWCRE_detrend_DJF = difference(SWCRE_DJF, 90)
LWCRE_detrend_DJF = difference(LWCRE_DJF, 90)
TGCLDLWP_detrend_DJF = difference(TGCLDLWP_DJF,90)
CLDTOT_detrend_DJF = difference(CLDTOT_DJF,90)

ICEFRAC_detrend_MAM = difference(ICEFRAC_MAM, 90)
SWCRE_detrend_MAM = difference(SWCRE_MAM, 90)
LWCRE_detrend_MAM = difference(LWCRE_MAM, 90)
TGCLDLWP_detrend_MAM = difference(TGCLDLWP_MAM,90)
CLDTOT_detrend_MAM = difference(CLDTOT_MAM,90)

###########################################################

# to remove noise from synoptic events, we smooth into 5 day chunks
# we use moving average function from earlier, now with a 5-day window.

ICEFRAC_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=ICEFRAC_detrend_DJF,window=5,mode='valid')
SWCRE_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=SWCRE_detrend_DJF,window=5,mode='valid')
LWCRE_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=LWCRE_detrend_DJF,window=5,mode='valid')
TGCLDLWP_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=TGCLDLWP_detrend_DJF,window=5,mode='valid')
CLDTOT_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=CLDTOT_detrend_DJF,window=5,mode='valid')

ICEFRAC_smooth_detrend_MAM = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=ICEFRAC_detrend_MAM,window=5,mode='valid')
SWCRE_smooth_detrend_MAM = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=SWCRE_detrend_MAM,window=5,mode='valid')
LWCRE_smooth_detrend_MAM = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=LWCRE_detrend_MAM,window=5,mode='valid')
TGCLDLWP_smooth_detrend_MAM = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=TGCLDLWP_detrend_MAM,window=5,mode='valid')
CLDTOT_smooth_detrend_MAM = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=CLDTOT_detrend_MAM,window=5,mode='valid')

###############################################################

# normalize data into anomalies
# function: subtract functions mean and divide by standard deviation:

def normalize(data):
    data = ((data - np.mean(data))/np.std(data))
    return data

# trim variables names to 'in', as they are now prepped to be VAR input.

ICEFRAC_in_DJF = normalize(ICEFRAC_smooth_detrend_DJF)
SWCRE_in_DJF = normalize(SWCRE_smooth_detrend_DJF)
LWCRE_in_DJF = normalize(LWCRE_smooth_detrend_DJF)
TGCLDLWP_in_DJF = normalize(TGCLDLWP_smooth_detrend_DJF)
CLDTOT_in_DJF = normalize(CLDTOT_smooth_detrend_DJF)

ICEFRAC_in_MAM = normalize(ICEFRAC_smooth_detrend_MAM)
SWCRE_in_MAM = normalize(SWCRE_smooth_detrend_MAM)
LWCRE_in_MAM = normalize(LWCRE_smooth_detrend_MAM)
TGCLDLWP_in_MAM = normalize(TGCLDLWP_smooth_detrend_MAM)
CLDTOT_in_MAM = normalize(CLDTOT_smooth_detrend_MAM)

#################################################################

# to fit a VAR model to our data, we use VAR class of the statsmodels package

# function: fit a VAR model for a 2d array of 1d time series variables
# input a 2d array containing time series for each var; shape is [n_obs,n_var]
# we select our maximum lag order using the bayesian information criterion
# first coefficient matrix in each VAR model is a constant; accounts
# for potential non-zero means in a given time series.

def VARmodel(dataset):
    VARmodel = sm.VAR(dataset)
    VARmodel_fit = VARmodel.fit(ic='bic',trend='c')
    return VARmodel_fit

# function: fit an individual VAR model for each lat/lon grid point in our data
# function assumes lat dimension = 60, lon dimension = 288
# grid_var3 is for VAR models with 3 variables.
# input smoothed, normalized, stationary time series for each var
# input icefrac_season raw data for 'icecheck'

def gridVAR_3(var1,var2,var3,icecheck):
    # initialize empty grid to be filled with VAR models
    gridmodels = np.zeros((60,288),dtype=object)
    # index over lat,lon grid points of each input var
    # create VAR datasets for fitting at each grid point
    # icecheck_index is raw, non-stationary data, and is not used in VAR model
    for i,j in np.ndindex(var1.shape[1:]) and np.ndindex(var2.shape[1:]) and \
    np.ndindex(var3.shape[1:]) and np.ndindex(icecheck.shape[1:]):
        x1 = var1[:,i,j]
        x2 = var2[:,i,j]
        x3 = var3[:,i,j]
        icecheck_index = icecheck[:,i,j]
        dataset = (np.array([x1,x2,x3])).T
        # only fit a VAR model if max ICEFRAC in a given time series
        # is above 5 percent
        if np.max(icecheck_index) > 5:
            gridmodels[i,j] = VARmodel(dataset)
    return gridmodels

# create VAR grids for summer and fall

VARgrid_summer = gridVAR_3\
(ICEFRAC_in_DJF,SWCRE_in_DJF,LWCRE_in_DJF,ICEFRAC_DJF)

VARgrid_fall = gridVAR_3\
(ICEFRAC_in_MAM,SWCRE_in_MAM,LWCRE_in_MAM,ICEFRAC_MAM)

######################################################################

# function: conduct granger causality test at one grid point
# predictands and predictors go by name 'y1','y2','y3' etc following gridVARs
# array order (see VAR grid computations to know what is what)
# returns binary depdending on whether null hypothesis is rejected or not.


def grangertest(model,predictand,predictor):
    test = model.test_causality(str(predictand),str(predictor),verbose=False)
    x=0
    if 'reject' in test.values():
        x = x+1
    else:
        x = x+2
    return x

# function: conduct granger causality tests over an entire VARgrid
# function assumes lat dimension = 60, lon dimension = 288
# assumes VARmodels include constant term so that df_model = 2 for VAR(1)
# input ingrid of VAR models, and variables to be tested
# output a grid of grangertest results where model meets requirements.

def grangergrid(ingrid,predictand,predictor):
    # initialize an empty lat/lon grid for granger binary results
    grangerresults = np.zeros((60,288))
    # loop over model in grid, test each model
    for i,j in np.ndindex(ingrid.shape):
        modelfit = ingrid[i,j]
        # only conduct causality tests if model is stable and contains
        # lag coefficients
        if modelfit != 0 and \
        modelfit.df_model > 1 and modelfit.is_stable() == True:
            result = grangertest(modelfit,str(predictand),str(predictor))
            grangerresults[i,j] = result
    return grangerresults

# function: get percent granger causality in a given granger grid.
# input ingrid of granger causality results
# output = float of percent granger causality

def causalitycount(ingrid):
    # initialize empty lists for percentage counts
    causelist = []
    nocauselist = []
    percentcausality = 0
    # loop over granger results grid, count results
    for i,j in np.ndindex(ingrid.shape):
        grangerresult = ingrid[i,j]
        if grangerresult == 1:
            causelist.append(grangerresult)
        elif grangerresult == 2:
            nocauselist.append(grangerresult)
    causecount = float(len(causelist))
    nocausecount = float(len(nocauselist))
    percentcausality = \
    ((causecount)/(causecount + nocausecount))*100
    return percentcausality


# grangergrid results for fall

grangergrid_fall_ICEtoSWCRE = grangergrid(VARgrid_fall,'y2','y1')
grangergrid_fall_SWCREtoICE = grangergrid(VARgrid_fall,'y1','y2')
grangergrid_fall_ICEtoLWCRE = grangergrid(VARgrid_fall,'y3','y1')
grangergrid_fall_LWCREtoICE = grangergrid(VARgrid_fall,'y1','y3')
grangergrid_fall_SWCREtoLWCRE = grangergrid(VARgrid_fall,'y3','y2')
grangergrid_fall_LWCREtoSWCRE = grangergrid(VARgrid_fall,'y2','y3')

# grangergrid results for summer

grangergrid_summer_ICEtoSWCRE = grangergrid(VARgrid_summer,'y2','y1')
grangergrid_summer_SWCREtoICE = grangergrid(VARgrid_summer,'y1','y2')
grangergrid_summer_ICEtoLWCRE = grangergrid(VARgrid_summer,'y3','y1')
grangergrid_summer_LWCREtoICE = grangergrid(VARgrid_summer,'y1','y3')
grangergrid_summer_SWCREtoLWCRE = grangergrid(VARgrid_summer,'y3','y2')
grangergrid_summer_LWCREtoSWCRE = grangergrid(VARgrid_summer,'y2','y3')

# causality counts for fall

ICEtoSWCREspace_fall = causalitycount(grangergrid_fall_ICEtoSWCRE)
SWCREtoICEspace_fall = causalitycount(grangergrid_fall_SWCREtoICE)
ICEtoLWCREspace_fall = causalitycount(grangergrid_fall_ICEtoLWCRE)
LWCREtoICEspace_fall = causalitycount(grangergrid_fall_LWCREtoICE)
SWCREtoLWCREspace_fall = causalitycount(grangergrid_fall_SWCREtoLWCRE)
LWCREtoSWCREspace_fall = causalitycount(grangergrid_fall_LWCREtoSWCRE)

# causality counts for summer

ICEtoSWCREspace_summer = causalitycount(grangergrid_summer_ICEtoSWCRE)
SWCREtoICEspace_summer = causalitycount(grangergrid_summer_SWCREtoICE)
ICEtoLWCREspace_summer = causalitycount(grangergrid_summer_ICEtoLWCRE)
LWCREtoICEspace_summer = causalitycount(grangergrid_summer_LWCREtoICE)
SWCREtoLWCREspace_summer = causalitycount(grangergrid_summer_SWCREtoLWCRE)
LWCREtoSWCREspace_summer = causalitycount(grangergrid_summer_LWCREtoSWCRE)

# plot percent causality info

print ICEtoSWCREspace_fall, ICEtoSWCREspace_summer
print SWCREtoICEspace_fall, SWCREtoICEspace_summer
print ICEtoLWCREspace_fall, ICEtoLWCREspace_summer
print LWCREtoICEspace_fall, LWCREtoICEspace_summer
print SWCREtoLWCREspace_fall, SWCREtoLWCREspace_summer
print LWCREtoSWCREspace_fall, LWCREtoSWCREspace_summer

 #########################################################

# plot grangergrid output on maps

# function: plot grid output on maps as contour data
# input grid, color levels (optional), projection
# shifts longitude data from 0 to 360 to -180 to 180
# returns contour map with spectral colors
# uses lat, lon from raw input dataset for coordinates
# resets lon to original state after doing shiftgrid

def mapplot(data,levels=None):
        lons = ICEFRAC_file.variables['lon'][:]
        lats = ICEFRAC_file.variables['lat'][0:60]
        map = Basemap(projection='splaea',boundinglat=-40, \
        lon_0=0,resolution='c', ax=ax)
        map.drawcoastlines()
        map.fillcontinents(color='#ffe2ab')
        # draw parallels and meridians.
        map.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
        map.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        # shift data so lons go from -180 to 180 instead of 0 to 360.
        data,lons = shiftgrid(180.,data,lons,start=False)
        llons, llats = np.meshgrid(lons, lats)
        x,y = map(llons,llats)
        cmap = plt.cm.get_cmap('spectral')
        cs = map.contourf(x,y,data,cmap=cmap,levels=levels)
        ## make a color bar
         # return lons to their original state
        lons = ICEFRAC_file.variables['lon'][:]
        return cs

# raw CRE plots
os.chdir('../draftfigures')
fig = plt.figure(figsize=[30,25])
fig.suptitle('ICEFRAC raw seasonal data',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin= 0
vmax= 100
levels = np.linspace(vmin,vmax,50)
ax.set_title('Summer', fontsize=30)
map1=mapplot(ICEFRAC_DJFplot,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin= 0
vmax= 100
levels = np.linspace(vmin,vmax,50)
ax.set_title('Fall', fontsize=30)
map2=mapplot(ICEFRAC_MAMplot,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('Ice fraction %',fontsize=30)
plt.savefig('ICEFRAC_rawdata_may29.png')
plt.close()





# Ice, SWCRE relationships

fig = plt.figure(figsize=[30,25])
fig.suptitle\
('Causality Relationships: Ice Fraction and Shortwave Cloud Radiative Effect',\
 fontsize=30)
ax = fig.add_subplot(2, 2, 1)
ax.set_title('Ice -> SWCRE, 70 percent causality', fontsize=20)
mapplot(grangergrid_summer_ICEtoSWCRE)
ax = fig.add_subplot(2,2,2)
ax.set_title('Ice -> SWCRE, 42 percent causality', fontsize=20)
mapplot(grangergrid_fall_ICEtoSWCRE)
ax = fig.add_subplot(2,2,3)
ax.set_title('SWCRE -> ICE, 21 percent causality', fontsize=20)
mapplot(grangergrid_summer_SWCREtoICE)
ax = fig.add_subplot(2,2,4)
ax.set_title('SWCRE -> ICE, 42 percent causality', fontsize=20)
mapplot(grangergrid_fall_SWCREtoICE)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.savefig('ICE_SWCRE_relationships_may24.png')
plt.close()

# Ice, LWCRE relationships

fig = plt.figure(figsize=[30,25])
fig.suptitle\
('Causality Relationships: Ice Fraction and Longwave Cloud Radiative Effect',\
 fontsize=30)
ax = fig.add_subplot(2, 2, 1)
ax.set_title('Ice -> LWCRE, 14 percent causality', fontsize=20)
mapplot(grangergrid_summer_ICEtoLWCRE)
ax = fig.add_subplot(2,2,2)
ax.set_title('Ice -> LWCRE, 35 percent causality', fontsize=20)
mapplot(grangergrid_fall_ICEtoLWCRE)
ax = fig.add_subplot(2,2,3)
ax.set_title('LWCRE -> ICE, 19 percent causality', fontsize=20)
mapplot(grangergrid_summer_LWCREtoICE)
ax = fig.add_subplot(2,2,4)
ax.set_title('LWCRE -> ICE, 41 percent causality', fontsize=20)
mapplot(grangergrid_fall_LWCREtoICE)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.savefig('ICE_LWCRE_relationships_may24.png')
plt.close()

# SWCRE, LWCRE relationships

fig = plt.figure(figsize=[30,25])
fig.suptitle\
('Causality Relationships: Shortwave and Longwave Cloud Radiative Effect',\
 fontsize=30)
ax = fig.add_subplot(2, 2, 1)
ax.set_title('SWCRE -> LWCRE, 13 percent causality', fontsize=20)
mapplot(grangergrid_summer_SWCREtoLWCRE)
ax = fig.add_subplot(2,2,2)
ax.set_title('SWCRE -> LWCRE, 10 percent causality', fontsize=20)
mapplot(grangergrid_fall_SWCREtoLWCRE)
ax = fig.add_subplot(2,2,3)
ax.set_title('LWCRE -> SWCRE, 13 percent causality', fontsize=20)
mapplot(grangergrid_summer_LWCREtoSWCRE)
ax = fig.add_subplot(2,2,4)
ax.set_title('LWCRE -> SWCRE, 17 percent causality', fontsize=20)
mapplot(grangergrid_fall_LWCREtoSWCRE)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.savefig('SWCRE_LWCRE_relationships_may24.png')
plt.close()

 ########################################################################

# function: analyze max lag order selection in each VAR model
# input VAR grid for a given season
# output, grid of lag order selections
# function assumes lat dimension = 60, lon dimension = 288
# assumes VARmodels include constant term so that df_model = 2 for VAR(1)

def maxlaggrid(ingrid,variablecount):
    # initialize an empty lat/lon grid for maxlag results
    maxlagresults = np.zeros((60,288))
    # loop over model in grid, test each model
    for i,j in np.ndindex(ingrid.shape):
        modelfit = ingrid[i,j]
        # only conduct measurement if model is stable
        # we subtract df_model by one since a constant is not a lag
        # we divide by the number of variables to get lag.
        if modelfit != 0 and modelfit.df_model > 1 and  \
        modelfit.is_stable() == True:
            results = (modelfit.df_model - 1)/int(variablecount)
            maxlagresults[i,j] = results
    return maxlagresults

# calculate maxlag grids

maxlagsfall = maxlaggrid(VARgrid_fall)
maxlagssummer = maxlaggrid(VARgrid_summer)

print np.max(maxlagsfall)
print np.max(maxlagssummer)

# plot maxlag grids

fig = plt.figure(figsize=[30,25])
fig.suptitle('Maximum lag-order selections per grid point\
, using Bayesian Information Criterion',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin=0.1
vmax=5
levels = np.linspace(vmin,vmax,6)
ax.set_title('Fall, (each lag unit = 5 days)', fontsize=30)
map1=mapplot(maxlagsfall,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin=0.1
vmax=5
levels = np.linspace(vmin,vmax,6)
ax.set_title('Summer (each lag unit = 5 days)', fontsize=30)
map2=mapplot(maxlagssummer,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('Maximum lag order per grid point',fontsize=30)
plt.savefig('Maxlags1maps_may24.png')
plt.close()

###################################################################

# coefficient analysis
# this function returns a grid of coefficients for user-specified variables
# at a user-specified lag period.
# pvalues are considered.
# assumes lon_dim = 288, lat_dim = 60, and that models include constants.

def coefficient_output(modelgrid,grangergrid,predictand,predictor,lag,varcount):
    # initialize an empty lat/lon grid for maxlag results
    coefresults = np.zeros((60,288))
    # loop over models in grids, test each grid point
    for i,j in np.ndindex(modelgrid.shape) and np.ndindex(grangergrid.shape):
        modelfit = modelgrid[i,j]
        grangerresult = grangergrid[i,j]
        # find relevant coefficients and pvalues
        # usual var requirements, except df_model threshold changes
        # according to lag order.
        # rows = predictors, where row 0 = constant term.
        # predictor input should start with 0,1,2, etc.
        # each lag should jump rows according to number of vars
        # lag-1 = 0 (no jumping), lag-2 = 1, etc.
        # columns = predictands, where variable 1 (ICEFRAC) equals zero.
        if modelfit != 0 and modelfit.df_model >= \
        (1 + ((lag+1)*int(varcount))) and modelfit.is_stable() == True:
            coef = modelfit.params\
            [int(1+int(predictor)+(int(lag)*int(varcount))),int(predictand)]
            pvalue = modelfit.pvalues\
            [int(1+int(predictor)+(int(lag)*int(varcount))),int(predictand)]
            # only output coefficient if causality exists and
            # value has 95% conf
            if pvalue <= .05 and grangerresult == 1:
                coefresults[i,j] = coef
    return coefresults

# function: find the mean and standard deviation of coefficients
# input coefgrid, output mean and standard deviation

def coef_stats(ingrid):
    coeflist = []
    for i,j in np.ndindex(ingrid.shape):
        coef = ingrid[i,j]
        if coef != 0:
            coeflist.append(coef)
    mean = np.mean(coeflist)
    stdev = np.std(coeflist)
    return mean,stdev


# plot coefs for ICEtoSWCRE_summer

ICEtoSWCRE_coefs1_summer = coefficient_output\
(VARgrid_summer,grangergrid_summer_ICEtoSWCRE,1,0,0,3)
ICEtoSWCRE_coefs2_summer = coefficient_output\
(VARgrid_summer,grangergrid_summer_ICEtoSWCRE,1,0,1,3)

print coef_stats(ICEtoSWCRE_coefs1_summer)
print coef_stats(ICEtoSWCRE_coefs2_summer)
print np.max(ICEtoSWCRE_coefs1_summer), np.max(ICEtoSWCRE_coefs2_summer)
print np.min(ICEtoSWCRE_coefs1_summer), np.min(ICEtoSWCRE_coefs2_summer)

fig = plt.figure(figsize=[30,25])
fig.suptitle('Ice -> SWCRE coefficients, summer',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-1 (5 days), mean = .25, stdev = .22', fontsize=30)
map1=mapplot(ICEtoSWCRE_coefs1_summer,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-2 (10 days), mean = -.27, stdev = .316', fontsize=30)
map2=mapplot(ICEtoSWCRE_coefs2_summer,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('coefficient',fontsize=30)
plt.savefig('ICE_SWCRE_coefs_summer.png')
plt.close()

# plot coefs for SWCREtoICE_summer

SWCREtoICE_coefs1_summer = coefficient_output\
(VARgrid_summer,grangergrid_summer_SWCREtoICE,0,1,0,3)
SWCREtoICE_coefs2_summer = coefficient_output\
(VARgrid_summer,grangergrid_summer_SWCREtoICE,0,1,1,3)

print coef_stats(SWCREtoICE_coefs1_summer)
print coef_stats(SWCREtoICE_coefs2_summer)
print np.max(SWCREtoICE_coefs1_summer), np.max(SWCREtoICE_coefs2_summer)
print np.min(SWCREtoICE_coefs1_summer), np.min(SWCREtoICE_coefs2_summer)


fig = plt.figure(figsize=[30,25])
fig.suptitle('SWCRE -> Ice coefficients, summer',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-1 (5 days), mean = .5, stdev = .7', fontsize=30)
map1=mapplot(SWCREtoICE_coefs1_summer,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin= -2
vmax= 2
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-2 (10 days), mean = .27, stdev = .56', fontsize=30)
map2=mapplot(SWCREtoICE_coefs2_summer,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('coefficient',fontsize=30)
plt.savefig('SWCRE_ICE_coefs_summer.png')
plt.close()

#plot coefs for ICEtoSWCRE_fall


ICEtoSWCRE_coefs1_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_ICEtoSWCRE,1,0,0,3)
ICEtoSWCRE_coefs2_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_ICEtoSWCRE,1,0,1,3)

print coef_stats(ICEtoSWCRE_coefs1_fall)
print coef_stats(ICEtoSWCRE_coefs2_fall)
print np.max(ICEtoSWCRE_coefs1_fall), np.max(ICEtoSWCRE_coefs2_fall)
print np.min(ICEtoSWCRE_coefs1_fall), np.min(ICEtoSWCRE_coefs2_fall)


fig = plt.figure(figsize=[30,25])
fig.suptitle('Ice -> SWCRE coefficients, fall',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin= -2
vmax= 2
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-1 (5 days), mean = .07, stdev = .29', fontsize=30)
map1=mapplot(ICEtoSWCRE_coefs1_fall,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin= -2
vmax= 2
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-2 (10 days), mean = .01, stdev = .40', fontsize=30)
map2=mapplot(ICEtoSWCRE_coefs2_fall,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('coefficient',fontsize=30)
plt.savefig('ICE_SWCRE_coefs_fall.png')
plt.close()

# plot coefs for SWCRE to Ice, fall

SWCREtoICE_coefs1_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_SWCREtoICE,0,1,0,3)
SWCREtoICE_coefs2_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_SWCREtoICE,0,1,1,3)

print coef_stats(SWCREtoICE_coefs1_fall)
print coef_stats(SWCREtoICE_coefs2_fall)
print np.max(SWCREtoICE_coefs1_fall), np.max(SWCREtoICE_coefs2_fall)
print np.min(SWCREtoICE_coefs1_fall), np.min(SWCREtoICE_coefs2_fall)

fig = plt.figure(figsize=[30,25])
fig.suptitle('SWCRE -> Ice coefficients, fall',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-1 (5 days), mean = 1.22, stdev = 1.04', fontsize=30)
map1=mapplot(SWCREtoICE_coefs1_fall,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin= -2
vmax= 2
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-2 (10 days), mean = .43, stdev = 1.72', fontsize=30)
map2=mapplot(SWCREtoICE_coefs2_fall,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('coefficient',fontsize=30)
plt.savefig('SWCRE_ICE_coefs_fall.png')
plt.close()

#plot coefs for ICEtoLWCRE_fall


ICEtoLWCRE_coefs1_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_ICEtoLWCRE,2,0,0,3)
ICEtoLWCRE_coefs2_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_ICEtoLWCRE,2,0,1,3)

print coef_stats(ICEtoLWCRE_coefs1_fall)
print coef_stats(ICEtoLWCRE_coefs2_fall)
print np.max(ICEtoLWCRE_coefs1_fall), np.max(ICEtoLWCRE_coefs2_fall)
print np.min(ICEtoLWCRE_coefs1_fall), np.min(ICEtoLWCRE_coefs2_fall)


fig = plt.figure(figsize=[30,25])
fig.suptitle('Ice -> LWCRE coefficients, fall',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-1 (5 days), mean = .13, stdev = .91', fontsize=30)
map1=mapplot(ICEtoLWCRE_coefs1_fall,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-2 (10 days), mean = -.10, stdev = .30', fontsize=30)
map2=mapplot(ICEtoLWCRE_coefs2_fall,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('coefficient',fontsize=30)
plt.savefig('ICE_LWCRE_coefs_fall.png')
plt.close()

# plot coefs for LWCRE to Ice, fall

LWCREtoICE_coefs1_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_LWCREtoICE,0,2,0,3)
LWCREtoICE_coefs2_fall = coefficient_output\
(VARgrid_fall,grangergrid_fall_LWCREtoICE,0,2,1,3)

print coef_stats(LWCREtoICE_coefs1_fall)
print coef_stats(LWCREtoICE_coefs2_fall)
print np.max(LWCREtoICE_coefs1_fall), np.max(LWCREtoICE_coefs2_fall)
print np.min(LWCREtoICE_coefs1_fall), np.min(LWCREtoICE_coefs2_fall)

fig = plt.figure(figsize=[30,25])
fig.suptitle('LWCRE -> Ice coefficients, fall',fontsize=30)
ax = fig.add_subplot(1,2,2)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-1 (5 days), mean = .31, stdev = .29', fontsize=30)
map1=mapplot(LWCREtoICE_coefs1_fall,levels=levels)
ax = fig.add_subplot(1, 2, 1)
vmin= -2.0
vmax= 2.0
levels = np.linspace(vmin,vmax,50)
ax.set_title('Lag-2 (10 days), mean = -.01, stdev = .82', fontsize=30)
map2=mapplot(LWCREtoICE_coefs2_fall,levels=levels)
plt.subplots_adjust\
(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
cb_ax = fig.add_axes()
cb = fig.colorbar(map2,ax=cb_ax,orientation='horizontal')
cb.set_label('coefficient',fontsize=30)
plt.savefig('LWCRE_ICE_coefs_fall.png')
plt.close()






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
        else:
            cs = map.contourf(x,y,data,cmap=cmap,levels=levels)
        ## make a color bar
        fig.colorbar\
        (cs, ax=ax,cmap=cmap, orientation='horizontal')
        # return lons to their original state
        lons = ICEFRAC_file.variables['lon'][:]
        return fig
