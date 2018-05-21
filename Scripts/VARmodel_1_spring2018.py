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

FSNS_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FSNS.15000101-15991231.nc')
FSNSC_file =\
('b.e11.B1850C5CN.f09_g16.005.cam.h1.FSNSC.15000101-15991231.nc')
FLNS_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.FLNS.15000101-15991231.nc')
FLNSC_file =\
('b.e11.B1850C5CN.f09_g16.005.cam.h1.FLNSC.15000101-15991231.nc')
ICEFRAC_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h1.ICEFRAC.15000101-15991231.nc')
TGCLDLWP_file =\
dt('b.e11.B1850C5CN.f09_g16.005.cam.h2.TGCLDLWP.1500010100Z-1510123118Z.nc')

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

# ICEFRAC units are converted from decimal to percent

ICEFRAC = ICEFRAC * 100

# clear sky radiation fluxes are subtracted from the net
# to obtain SW and LW cloud radiative effects

SWCRE = FSNS - FSNSC
LWCRE = FLNS - FLNSC

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

# isolate DJF for all variables
ICEFRAC_DJF = seasonselect(ICEFRAC,'DJF')
FSNS_DJF = seasonselect(FSNS,'DJF')
TGCLDLWP_DJF = seasonselect(TGCLDLWP,'DJF')



plt.subplot(2,1,1)
plt.plot(ICEFRAC_DJF[:,27,100],'b')
plt.ylabel('coverage')
plt.title('ICE fraction')
plt.subplot(2,1,2)
plt.plot(FSNS_DJF[:,27,100],'r')
plt.ylabel('w/sq. m')
plt.title('Net shortwave radiation at the surface')
os.chdir('../draftfigures')
plt.savefig('seasonalplots_may17.png')
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
    diff = np.array(diff)
    return diff

# for seasonally separated data, harmonic = 90 days

ICEFRAC_detrend_DJF = difference(ICEFRAC_DJF, 90)
FSNS_detrend_DJF = difference(FSNS_DJF, 90)
TGCLDLWP_detrend_DJF = difference(TGCLDLWP_DJF,90)

###########################################################
# to remove noise from synoptic events, we smooth into 5 day chunks
# we use moving average function from earlier, now with a 5-day window.

ICEFRAC_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=ICEFRAC_detrend_DJF,window=5,mode='valid')
FSNS_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=FSNS_detrend_DJF,window=5,mode='valid')
TGCLDLWP_smooth_detrend_DJF = np.apply_along_axis\
(func1d=movingaverage,axis=0,arr=TGCLDLWP_detrend_DJF,window=5,mode='valid')

###########################################################
# normalize data into anomalies
# function: subtract functions mean and divide by standard deviation:

def normalize(data):
    data = ((data - np.mean(data))/np.std(data))
    return data

ICEFRAC_norm_smooth_detrend_DJF = normalize(ICEFRAC_smooth_detrend_DJF)
FSNS_norm_smooth_detrend_DJF = normalize(FSNS_smooth_detrend_DJF)
TGCLDLWP_norm_smooth_detrend_DJF = normalize(TGCLDLWP_smooth_detrend_DJF)


###########################################################
# conduct granger causality test, main implementation

# function: create a VARmodel, input is 2d array of shape (n_obs,n_var)
# this function is applied to one grid point at a time.

def VARmodel(dataset):
    VARmodel = sm.VAR(dataset)
    VARmodel_fit = VARmodel.fit(ic='bic',trend='c')
    return VARmodel_fit


def maxlagoutput(ingrid):
    for i,j in np.ndindex(ICEFRAC_in.shape[1:]) and \
    np.ndindex(FSNS_in.shape[1:]) and np.ndindex(TGCLDLWP_in.shape[1:])\
    and np.ndindex(ICEFRAC_check.shape):
        x1 = ICEFRAC_in[:,i,j]
        x2 = FSNS_in[:,i,j]
        x3 = TGCLDLWP_in[:,i,j]
        icecheck = ICEFRAC_check[i,j]
        dataset = (np.array([x1,x2,x3])).T
        # don't fit a VARmodel if ICEFRAC is at or near constant
        #(i.e. approx. 0 throughout time)
        if icecheck < 5:
            result = 0
        else:
            modelfit = VARmodel(dataset)
            # if AIC chooses a lag-0 model, avoid grangertest and output the
            #grid point where it occurs to see what went wrong
            if modelfit.df_model==1:
                result = 0
                print (i,j),'no-lags'
            elif modelfit.is_stable() != True:
                result = 0
                print (i,j),'unstable'
            else:
                lagresults = (modelfit.df_model - 1)
                ingrid[i,j] = lagresults
    return ingrid

ICEFRAC_in = ICEFRAC_norm_smooth_detrend_DJF
FSNS_in = FSNS_norm_smooth_detrend_DJF
TGCLDLWP_in = TGCLDLWP_norm_smooth_detrend_DJF
ICEFRAC_check = np.max(ICEFRAC_DJF,axis=0)

maxlaggrid = ICEFRAC_in*0
maxlaggrid = np.mean(maxlaggrid,axis=0)
maxlaggrid = maxlagoutput(maxlaggrid)

def grangertest(model,predictand,predictor):
    test = model.test_causality(str(predictand),str(predictor),verbose=False)
    x=0
    if 'reject' in test.values():
        x = x+1
    else:
        x = x+2
    return x

def fullgridgranger(predictand,predictor):
    # initialize 2d lat/lon array
    # with dimensions equal to input variable(s) lat/lon
    # float values = 0 to start
    predictand = str(predictand)
    predictor = str(predictor)
    grangergrid = ICEFRAC_in*0
    grangergrid = np.mean(grangergrid,axis=0)
    causelist = []
    nocauselist = []
    for i,j in np.ndindex(ICEFRAC_in.shape[1:]) and \
    np.ndindex(FSNS_in.shape[1:]) and np.ndindex(TGCLDLWP_in.shape[1:])\
    and np.ndindex(ICEFRAC_check.shape):
        x1 = ICEFRAC_in[:,i,j]
        x2 = FSNS_in[:,i,j]
        x3 = TGCLDLWP_in[:,i,j]
        icecheck = ICEFRAC_check[i,j]
        dataset = (np.array([x1,x2,x3])).T
        # don't fit a VARmodel if ICEFRAC is at or near constant
        #(i.e. approx. 0 throughout time)
        if icecheck < 5:
            result = 0
        else:
            modelfit = VARmodel(dataset)
            # if AIC chooses a lag-0 model, avoid grangertest and output the
            #grid point where it occurs to see what went wrong
            if modelfit.df_model==1:
                result = 0
            elif modelfit.is_stable() != True:
                result = 0
            else:
                grangerresult = \
                grangertest(modelfit,predictand,predictor)
                grangergrid[i,j]=grangerresult
                if grangerresult == 1:
                    causelist.append(grangerresult)
                elif grangerresult == 2:
                    nocauselist.append(grangerresult)
                #causelist = np.array(causelist)
                #nocauselist = np.array(nocauselist)
                causecount = float(len(causelist))
                nocausecount = float(len(nocauselist))
                percentcausality = \
                ((causecount)/(causecount + nocausecount))*100
    return grangergrid,percentcausality


grangergrid_ICEtoFSNS,ICEtoFSNSspace = fullgridgranger('y2','y1')
grangergrid_FSNStoICE,FSNStoICEspace = fullgridgranger('y1','y2')
grangergrid_ICEtoCLDLWP,ICEtoCLDLWPspace = fullgridgranger('y3','y1')
grangergrid_CLDLWPtoICE,CLDLWPtoICEspace = fullgridgranger('y1','y3')
grangergrid_FSNStoCLDLWP,FSNStoCLDLWPspace = fullgridgranger('y3','y2')
grangergrid_CLDLWPtoFSNS,CLDLWPtoFSNSspace = fullgridgranger('y2','y3')




#function: fit VARmodels for each lat/lon grid point
# function syntax must be adjusted depending on number of input VARS
# icecheck is a 2d variable used as a condition for building a VAR model
# output is a lat/lon grid with a VAR model at each grid point
# assumes lat dimension = 60, lon dimension = 288

def gridVARs(var1,var2,var3,icecheck):
    # initialize empty grid to be filled with VAR models
    gridmodels = np.zeros((60,288),dtype=object)
    # index over lat,lon grid points of each input var and create dataset
    # note that icecheck has no time dimension
    for i,j in np.ndindex(var1.shape[:1]) and np.ndindex(var2.shape[:1]) and np.ndindex(var3.shape[:1]) and np.ndindex(icecheck.shape):
        x1 = var1[:,i,j]
        x2 = var1[:,i,j]
        x3 = var1[:,i,j]
        icecheck_index = icecheck[i,j]
        dataset = (np.array([x1,x2,x3])).T
        # only create a VAR model if max ICEFRAC in a given time series
        # is above 5 percent
        if icecheck_index > .05:
            gridmodels[i,j] = VARmodel(dataset)
    return gridmodels


# outputs coefficients to grid at a given lag if p-value less than .05
# input predictands as integers (0,1,2,3....n)

def coefficient_output(predictand,predictor,grangergrid,lag):
    coefgrid = ICEFRAC_in*0
    coefgrid = np.mean(coefgrid,axis=0)
    for i,j in np.ndindex(ICEFRAC_in.shape[1:]) and \
    np.ndindex(FSNS_in.shape[1:]) and np.ndindex(TGCLDLWP_in.shape[1:])\
    and np.ndindex(ICEFRAC_check.shape):
        x1 = ICEFRAC_in[:,i,j]
        x2 = FSNS_in[:,i,j]
        x3 = TGCLDLWP_in[:,i,j]
        icecheck = ICEFRAC_check[i,j]
        dataset = (np.array([x1,x2,x3])).T
        # don't fit a VARmodel if ICEFRAC is at or near constant
        #(i.e. approx. 0 throughout time)
        if icecheck < 5:
            result = 0
        else:
            modelfit = VARmodel(dataset)
            # if AIC chooses a lag-0 model, avoid analysis and output the
            #grid point where it occurs to see what went wrong
            if modelfit.df_model <= lag+3:
                result = 0
            elif modelfit.is_stable() != True:
                result = 0
            else:
                coef = modelfit.params\
                [int((1+int(predictor)+(int(lag)*3))),int(predictand)]
                pvalue = modelfit.pvalues\
                [int((1+int(predictor)+(int(lag)*3))),int(predictand)]
                if pvalue <= .05 and grangergrid[i,j]==1:
                    coefgrid[i,j] = coef
    return coefgrid

ICEtoFSNScoef1 = coefficient_output(1,0,grangergrid_ICEtoFSNS,0)
ICEtoFSNScoef2 = coefficient_output(1,0,grangergrid_ICEtoFSNS,1)
ICEtoFSNScoef3 = coefficient_output(1,0,grangergrid_ICEtoFSNS,3)


###########################################################
# plot desired grids on map
# function, input grid to be plotted, title

def mapplot(data,title,levels=None,Normalize=False,fillcontinents=False):
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
        if Normalize is True:
            norm = c.LogNorm(vmin=np.min(data),vmax=np.max(data))
            cs = map.contourf(x,y,data,cmap=cmap,levels=levels,\
            norm=norm,shading='interp')
        else:
            cs = map.contourf(x,y,data,cmap=cmap,levels=levels)
        ## make a color bar
        fig.colorbar\
        (cs, ax=ax,cmap=cmap, orientation='horizontal')
        # return lons to their original state
        lons = ICEFRAC_file.variables['lon'][:]
        return fig

# make Basemap
os.chdir('../draftfigures')
figure_maxlag = mapplot(maxlaggrid,\
'Model lag-order by gridpoint (lag-1=5 days), method=bic',\
levels=[0,1,2,3,4,5,6,7,8,9,10],fillcontinents=True)
figure_maxlag.savefig('maxlags_bic_may17.png')
plt.close()

##########################################################
grangerfig = mapplot(grangergrid_ICEtoFSNS,\
'ICE -> FSNS, DJF',fillcontinents=True)
grangerfig.savefig('grangergrid_ICEtoFSNS_may17.png')
plt.close()

grangerfig = mapplot(grangergrid_FSNStoICE,\
'FSNS -> ICE, DJF',fillcontinents=True)
grangerfig.savefig('grangergrid_FSNStoICE_may17.png')
plt.close()

grangerfig = mapplot(grangergrid_ICEtoCLDLWP,\
'ICE -> CLDLWP, DJF',fillcontinents=True)
grangerfig.savefig('grangergrid_ICEtoCLDLWP_may17.png')
plt.close()

grangerfig = mapplot(grangergrid_CLDLWPtoICE,\
'CLDLWP -> ICE, DJF',fillcontinents=True)
grangerfig.savefig('grangergrid_CLDLWPtoICE_may17.png')
plt.close()

grangerfig = mapplot(grangergrid_FSNStoCLDLWP,\
'FSNS -> CLDLWP, DJF',fillcontinents=True)
grangerfig.savefig('grangergrid_FSNStoCLDLWP_may17.png')
plt.close()

grangerfig = mapplot(grangergrid_CLDLWPtoFSNS,\
'CLDLWP -> FSNS, DJF',fillcontinents=True)
grangerfig.savefig('grangergrid_CLDLWPtoFSNS_may17.png')
plt.close()

##########################################################

os.chdir('../draftfigures')
figure_coef = mapplot(ICEtoFSNScoef1,\
'ICE -> FSNS, DJF (lag-1 coefficients)',\
levels=[-.5,-.1,-.05,0,.05,.1,.5,],fillcontinents=True)
figure_coef.savefig('ICEtoFSNScoef1.png')
plt.close()
#make basemap 2
os.chdir('../draftfigures')
figure_coef = mapplot(ICEtoFSNScoef2,\
'ICE -> FSNS, DJF (lag-2 coefficients)',\
levels=[-.5,-.1,-.05,0,.05,.1,.5,],fillcontinents=True)
figure_coef.savefig('ICEtoFSNScoef2.png')
plt.close()
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
'ICEFRAC -> Net incoming SR (surface), DJF',fillcontinents=True)
figure_granger.savefig('granger1_may14.png')
plt.close()

#basemap 6
os.chdir('../draftfigures')
figure_maxlag = mapplot(maxlaggrid,\
'Model lag-order by gridpoint (lag-1=5 days)',\
levels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,\
26,27,28],fillcontinents=True)
figure_maxlag.savefig('maxlag1_may14.png')
plt.close()

#basemap 7
os.chdir('../draftfigures')
figure_maxlag = mapplot(maxlaggrid,\
'Model lag-order by gridpoint (5-day units),selection criteria = bic',\
levels=[1,2,3,4,5,6,7,8,9,10,11,12],fillcontinents=True)
figure_maxlag.savefig('maxlagbic_may14.png')
plt.close()


#basemap 11
os.chdir('../draftfigures')
figure_granger = mapplot(coefgrid2,\
'Net incoming SR (surface) -> ICEFRAC, DJF (lag-2 coefficients, p=.05)',\
levels=[-1,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1,-.05,-.01,0,.01,.05,.1,.2,.3],\
fillcontinents=True)
figure_granger.savefig('coef2_bic_may15.png')
plt.close()

os.chdir('../draftfigures')
figure_granger = mapplot(coefgrid3,\
'ICEFRAC -> Net incoming SR (surface), DJF (lag-1 coefficients, p=.05)',\
fillcontinents=True)
figure_granger.savefig('coef3_bic_may15.png')
plt.close()

os.chdir('../draftfigures')
figure_granger = mapplot(coefgrid4,\
'ICEFRAC -> Net incoming SR (surface), DJF (lag-2 coefficients, p=.05)',\
fillcontinents=True)
figure_granger.savefig('coef4_bic_may15.png')
plt.close()
