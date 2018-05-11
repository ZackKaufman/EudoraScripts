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

# import LAT and LON data from one variable for map generation
lons = ICEFRAC_file.variables['lon'][:]
lats = ICEFRAC_file.variables['lat'][:]
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

# shorten the names of the input variables. values remain unchanged.
ICEFRAC_in = ICEFRAC_norm_smooth_detrend_DJF[:,0:30,0:5]
FSNS_in = FSNS_norm_smooth_detrend_DJF[:,0:30,0:5]

# initialize 2d lat/lon array with values = 0 and dim_lengths = input variable
# values will be updated by granger causality tests
grangergrid = ICEFRAC_in*0
grangergrid = np.mean(grangergrid,axis=0)

# create VAR model and conduct granger test at each lat/lon grid point.
# fill grangergrid with granger results binary. All values should = 1(T) or 2(F)
# edit 2nd to last line in loop to change vars being considered for causality
for i,j in np.ndindex(ICEFRAC_in.shape[1:]) and np.ndindex(FSNS_in.shape[1:]):
    y1 = ICEFRAC_in[:,i,j]
    y2 = FSNS_in[:,i,j]
    # the following calculation line adds small amounts of random noise
    # this eliminates the singular matrix problem when ICEFRAC is nonexistent
    dataset = [y1,y2]+.00000001*np.random.rand(2,810)
    dataset = np.array(dataset)
    dataset = dataset.T
    ICEFRAC_FSNS_VARmodel = VARmodel(dataset)
    result = grangertest(ICEFRAC_FSNS_VARmodel,'y1','y2')
    grangergrid[i,j] = result

###########################################################
# plot granger grid on map

# trim lons/lats to = grangergrid dimensions
lons = lons[0:5]
lats = lats[0:30]


fig = plt.figure(figsize=[12,15])  # a new figure window
ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
ax.set_title('Net incoming SR -> ICEFRAC?', fontsize=14)

map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=-40,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c', ax=ax)

map.drawcoastlines()
map.fillcontinents(color='#ffe2ab')
# draw parallels and meridians.
map.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
map.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

# shift data so lons go from -180 to 180 instead of 0 to 360.
grangergrid,lons = shiftgrid(180.,sic,lons,start=False)
llons, llats = np.meshgrid(lons, lats)
x,y = map(llons,llats)
# make a color map of fixed colors
cmap = c.ListedColormap(['#00004c','#000080','#0000b3','#0000e6','#0026ff','#004cff',
                             '#0073ff','#0099ff','#00c0ff','#00d900','#33f3ff','#73ffff','#c0ffff',
                             (0,0,0,0),
                             '#ffff00','#ffe600','#ffcc00','#ffb300','#ff9900','#ff8000','#ff6600',
                             '#ff4c00','#ff2600','#e60000','#b30000','#800000','#4c0000'])
bounds=[0,1,2]
norm = c.BoundaryNorm(bounds, ncolors=cmap.N) # cmap.N gives the number of colors of your palette

cs = map.contourf(x,y,grangergrid, cmap=cmap, norm=norm, levels=bounds,shading='interp')

## make a color bar
fig.colorbar(cs, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, ax=ax, orientation='horizontal')
fig.savefig('grangertestmap_may11.png')
plt.close()
