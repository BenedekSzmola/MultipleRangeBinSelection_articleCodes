"""
This file "createFigs.py" creates the figures of the article
Keep in mind that there are 2 figures (Fig 3 & 4) which are composed of multiple sub figures. This script will create the figures separately, the final figures were put 
    together using PowerPoint
Figure 1 was created in PowerPoint
"""

### Import packages
import numpy as np
from scipy import signal,sparse
import copy
import csv
import matplotlib.pyplot as plt
import platform
from ripser import ripser
from readRecordingData import *

import helperFunctions as uS
##############################################################################################################

def sec2hourMinSec(timeInput):
    hour = timeInput // 3600
    minRem = timeInput % 3600
    min = minRem // 60
    sec = minRem % 60

    return hour,min,sec

def makeSecTax2Hour(timeVec,ax2conv):
    
    newStamps = []
    newStampLabels = []
    roundStart = 1800 * np.ceil(timeVec[0]/1800).astype("int")
    for stampi in range(roundStart,timeVec[-1],1800):
        stampiH,stampiM,_ = sec2hourMinSec(stampi)
        newStamps.append(stampi)
        newStampLabels.append(f'{stampiH:d}:{stampiM:02d}')

    ax2conv.set_xticks(newStamps,newStampLabels)
    ax2conv.set_xlabel("Time [hours:minutes]")
    
    return

if platform.system() == "Linux":
    path2saveFigs = "figures//"
elif platform.system() == "Windows":
    path2saveFigs = "figures\\"

if platform.system() == "Linux":
    path2saveFiles = "processedData//"
elif platform.system() == "Windows":
    path2saveFiles = "processedData\\" 


with open('radarSettings.csv') as csv_file:
    reader = csv.reader(csv_file)
    radarSettings = dict(reader)

radarSettings = {key:float(value) for key,value in radarSettings.items()}
for key,value in radarSettings.items():
    if value.is_integer():
        radarSettings[key] = int(value)

####################################################################################################
####################################################################################################

### Figure 2
# data from Subj10, timestart:10550
# Loading the data for the figure
recID = "10"
filePath = f"placeholder_subj{recID}" # insert your filepath here
file_info,radar_var,synchro_info,measurement_data = readSaveFile(file_name=filePath)

radar_idx = 38
radar_srate = synchro_info['Effective sampling frequency given by xdf.load() (Radar_1, Radar_2, Radar_5)'][radar_idx-38]

epochInputDict = {
    'epochStarts': np.array([10550]) + measurement_data[radar_idx][0][0]
}

timestampEpochs,phaseEpochs,_,_,_ = readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,radar_srate,15,epochInput=epochInputDict,useHann=True,chirp=np.arange(radarSettings['radar_loop_num']),chirpSumMethod="median",doUnwrap=True,norm01=False,removeDC=True)
# end of loading the data


bins2check = np.arange(9,61)

currRadarTime = timestampEpochs[0]
currPhase = phaseEpochs[0]
data4RangeTime = copy.deepcopy(currPhase[bins2check,:])

data4RangeTime = data4RangeTime - np.mean(data4RangeTime,axis=1,keepdims=True)

k = int(.5*radar_srate)
data4RangeTime = signal.filtfilt(np.ones(k)/k,1,data4RangeTime)

plt.rc('font',size=12)
fig,ax = plt.subplots(1,1,figsize=(14,8),layout='constrained',dpi=400)

imi1 = ax.imshow(data4RangeTime, origin='lower', extent=[0, currRadarTime[-1]-currRadarTime[0],\
        bins2check[0]*.05-.025, bins2check[-1]*.05+.025], aspect='auto', interpolation='none',cmap='viridis', vmin=-.1,vmax=.1)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance from radar [m]")
ax.set_title(f"Filtered Radar Phase Signals - Range Bins {bins2check[0]} - {bins2check[-1]} ({.05*bins2check[0]:.2f} - {.05*bins2check[-1]:.2f} m)")

plt.tight_layout()

fig.subplots_adjust(right=0.97)

cbar_ax0 = fig.add_axes([0.975, ax.get_position().y0, 0.02, ax.get_position().height])
fig.colorbar(imi1, label="Displacement [mm]", cax=cbar_ax0)

plt.savefig(path2saveFigs + f'fig2.jpg',dpi=1200,bbox_inches="tight")

plt.show()
plt.rc('font',size=10)
####################################################################################################

### Figure 3 - data from subject 09 timestart 16550 range bin 40
# Panel A - timeseries radar + psg

# Loading the data for the figure
recID = "09"
filePath = f"placeholder_subj{recID}" # insert your filepath here
file_info,radar_var,synchro_info,measurement_data = readSaveFile(file_name=filePath)

radar_idx = 38
radar_srate = synchro_info['Effective sampling frequency given by xdf.load() (Radar_1, Radar_2, Radar_5)'][radar_idx-38]

epochInputDict = {
    'epochStarts': np.array([16550]) + measurement_data[radar_idx][0][0]
}

timestampEpochs,phaseEpochs,_,_,_ = readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,radar_srate,15,epochInput=epochInputDict,useHann=True,chirp=np.arange(radarSettings['radar_loop_num']),chirpSumMethod="median",doUnwrap=True,norm01=True,removeDC=True)

Thorax_idx = uS.getSensorIdx(file_info['Measurement Data Format'],"Thorax")
Thorax_name = file_info['Measurement Data Format'][Thorax_idx]
Thorax_srate = int(float(file_info['Measurement Data Sampling Rate'][Thorax_idx]))

ThoraxTimestampsFull,ThoraxDataFull = readFullRefData(measurement_data,Thorax_idx)
ThoraxTimestampEpochs,ThoraxEpochs,ThoraxEpochStartTimestamps,ThoraxTimeStarts = makeRefDataEpochs(ThoraxTimestampsFull,ThoraxDataFull,Thorax_srate,15,epochInputDict,norm01=True)
# end of loading the data

bin2use = 40

currRadarTime = timestampEpochs[0]
currPhase = uS.normDataTo_0_1(phaseEpochs[0])
currThoraxTime = ThoraxTimestampEpochs[0]
currThorax = ThoraxEpochs[0]

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(14,6))

ax0 = fig.add_subplot(1,1,1)

ax0.plot(currRadarTime-currRadarTime[0],currPhase[bin2use,:],'k',label='Radar phase')
ax0.plot(currThoraxTime-currThoraxTime[0],currThorax,'r',alpha=.5,label='Thorax belt')
ax0.set_title(f'Raw Radar Phase Data and PSG Reference')
ax0.set_ylabel("Normalized displacement [a.u.]")
ax0.legend(loc=1)
ax0.set_xlabel("Time [s]")

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig3_a.jpg',dpi=1200,bbox_inches="tight")

plt.show()
plt.rcParams.update({'font.size': 10})

######################################################
# Panel B - time delay embedding

embedDelay = 10
embedDim = 3
timeDelayEmbed = uS.createDelayVector(currPhase[bin2use,:],embedDelay,embedDim)

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1.plot3D(*timeDelayEmbed[:,:3].T, 'o-', markersize=8, fillstyle='none')

ax1.set_title(f'Time Delay Embedding of Radar Phase Signal (m=3, τ=10)')
ax1.set_xlabel('x(t)')
ax1.set_ylabel(f'x(t+{embedDelay})')
ax1.set_zlabel(f'x(t+{2*embedDelay})')

ax1.set_box_aspect(None, zoom=.9)

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig3_b.jpg',dpi=1200,bbox_inches="tight")

plt.show()
plt.rcParams.update({'font.size': 10})

######################################################
# Panel C  - persistence diagram

diagrams = ripser(timeDelayEmbed)['dgms']
diagrams[0] = diagrams[0][:-1,:]

plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(8,8))

plt.plot(diagrams[0][:,0],np.diff(diagrams[0],axis=1),'bo',markersize=6,label="$H_{0}$")
plt.plot(diagrams[1][:,0],np.diff(diagrams[1],axis=1),'go',markersize=6,label="$H_{1}$")
plt.xlabel("Birth time")
plt.ylabel("Lifespan")
plt.title('Persistence Diagram')
plt.legend()

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig3_c.jpg',dpi=1200,bbox_inches="tight")

plt.show()
plt.rcParams.update({'font.size': 10})

####################################################################################################

### Figure 4
# Panel A - segmentwise colored timeseries
N = len(currPhase[bin2use,:])
b,a = signal.butter(N=2,Wn=[.1,.4],btype="bandpass",fs=radar_srate)
x = uS.normDataTo_0_1(signal.filtfilt(b,a,currPhase[bin2use,:]))
t = copy.deepcopy(currRadarTime)
# Add edges between adjacent points in the time series, with the "distance"
# along the edge equal to the max value of the points it connects
I = np.arange(N-1)
J = np.arange(1, N)
V = np.maximum(x[0:-1], x[1::])
# Add vertex birth times along the diagonal of the distance matrix
I = np.concatenate((I, np.arange(N)))
J = np.concatenate((J, np.arange(N)))
V = np.concatenate((V, x))
#Create the sparse distance matrix
D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]

dgm0 = dgm0[dgm0[:, 1]-dgm0[:, 0] > 1e-3, :]
allgrid = np.unique(dgm0.flatten())
allgrid = allgrid[allgrid < np.inf]
xs = np.unique(dgm0[:, 0])
ys = np.unique(dgm0[:, 1])
ys = ys[ys < np.inf]

dgm0 = dgm0[:-1,:]

allgrid_ = copy.deepcopy(allgrid)
xs_ = copy.deepcopy(xs)
ys_ = copy.deepcopy(ys)

plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(12, 8))


ylims = [-.1, 1.1]
plt.plot(t-t[0], x, 'b', linewidth=2)
ax = plt.gca()
ax.set_yticks(allgrid,[f'{i:.2f}' for i in allgrid],rotation=0)
plt.ylim(ylims)
plt.grid(linewidth=1, linestyle='--', axis='y')
plt.title("Bandpass-filtered Radar Phase Signal")
plt.ylabel("Normalized displacement [a.u.]")
plt.xlabel("Time [s]")


# do the segmentwise coloring
underThrInds = np.where(x <= .371)[0]
underThrInds = underThrInds[uS.intervalExtractor(underThrInds)[0]]
for i in range(underThrInds.shape[0]):
    if i==0:
        lbl = "Sections below y=0.37"
    else:
        lbl = None

    plt.plot(t[underThrInds[i,0]:underThrInds[i,1]+1] - t[0], x[underThrInds[i,0]:underThrInds[i,1]+1],'r',linewidth=12,label=lbl)

plt.plot(t[underThrInds[1,0]:underThrInds[2,1]+1] - t[0], x[underThrInds[1,0]:underThrInds[2,1]+1],'#FFFF33',linewidth=4,label="Merged section at y=0.71")


plt.legend()
plt.tight_layout()

plt.savefig(path2saveFigs + f'fig4_a.jpg',dpi=1200,bbox_inches="tight")

plt.show()


######################################################
# Panel B  - persistence diagram
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(8, 8))

ax = plt.gca()
plt.ylim([.5, 1.1])
plt.xlim([-.1, .7])
ax.set_yticks(ys_,[f'{i:.2f}' for i in ys_])
ax.set_xticks(xs_,[f'{i:.2f}' for i in xs_],rotation=45)
plt.ylim(ylims)
plt.grid(linewidth=1, linestyle='--')

plt.plot(dgm0[:,0],dgm0[:,1],'bo',markersize=6,label="$H_{0}$")
plt.plot([0,1],[0,1],'k-',label="Lifetime = 0")
plt.xlabel("Birth time")
plt.ylabel("Death time")
plt.title('Persistence Diagram')
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig4_b.jpg',dpi=1200,bbox_inches="tight")

plt.show()

plt.rcParams.update({'font.size': 10})

####################################################################################################

### Figure 5 - this is created by taking part of two 3 part subplots, but only the 3rd subplot is used 
# Breathing range profile
# loading the save files for the figures
with open(path2saveFiles + f"subj08_BR_chirpsall_procData.pkl", 'rb') as fp:
    resultsDict = pickle.load(fp)

bestBinsBR = resultsDict['selectedBins']
timeStarts = resultsDict['timeStarts']
timeWinLen = resultsDict['epochLen']
bins2check = resultsDict['bins2check']
# end of loading the save files

plt.rc('font',size=12)
fig,ax = plt.subplots(3,1,sharex=True,figsize=(16,10))
fig.tight_layout()

cax = ax[2].matshow(bestBinsBR, extent=[timeStarts[0]-timeWinLen//2, timeStarts[-1]+timeWinLen//2, bins2check[0]*.05-.025, bins2check[-1]*.05+.025], aspect='auto', origin='lower', cmap='plasma')
ax[2].xaxis.set_ticks_position('bottom')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel("Distance from radar [m]")
ax[2].set_title("Radar Breathing Range Profile")

makeSecTax2Hour(timeStarts,ax[2])

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig5_a.jpg',dpi=1200,bbox_inches="tight")

plt.show()

plt.rc('font',size=10)

# Heartbeat range profile
# loading the save files for the figures
with open(path2saveFiles + f"subj08_HR_chirpsall_procData.pkl", 'rb') as fp:
    resultsDict = pickle.load(fp)

bestBinsHR = resultsDict['selectedBins']
timeStarts = resultsDict['timeStarts']
timeWinLen = resultsDict['epochLen']
bins2check = resultsDict['bins2check']
# end of loading the save files

plt.rc('font',size=12)
fig,ax = plt.subplots(3,1,sharex=True,figsize=(16,10))
fig.tight_layout()

cax = ax[2].matshow(bestBinsHR, extent=[timeStarts[0]-timeWinLen//2, timeStarts[-1]+timeWinLen//2, bins2check[0]*.05-.025, bins2check[-1]*.05+.025], aspect='auto', origin='lower', cmap='plasma')
ax[2].xaxis.set_ticks_position('bottom')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel("Distance from radar [m]")
ax[2].set_title("Radar Heartbeat Range Profile")

makeSecTax2Hour(timeStarts,ax[2])

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig5_b.jpg',dpi=1200,bbox_inches="tight")

plt.show()

plt.rc('font',size=10)

####################################################################################################

### Figure 6 - breathing results overnight
# loading the save files for the figures
with open(path2saveFiles + f"subj10_BR_chirpsall_procData.pkl", 'rb') as fp:
    resultsDict = pickle.load(fp)

bestBinsBR =    resultsDict['selectedBins']
timeStarts =    resultsDict['timeStarts']
timeWinLen =    resultsDict['epochLen']
bins2check =    resultsDict['bins2check']
psgDataBR  =    resultsDict['psgVitalRates']
radarDataBR =   resultsDict['medianRadarVitalRates']

winsWithBoth = len(np.where((~np.isnan(radarDataBR)) & (~np.isnan(psgDataBR)))[0])
mape = (1/winsWithBoth) * np.nansum(np.abs(psgDataBR - radarDataBR) / np.abs(psgDataBR)) * 100
# end of loading the save files

plt.rc('font',size=12)
fig,ax = plt.subplots(3,1,sharex=True,figsize=(16,10))
fig.tight_layout()

ax[0].set_title("Breathing Rate Computed from Radar and PSG")

ax[0].plot(timeStarts, psgDataBR, 'rs', label='PSG', alpha=.75, markersize=3, fillstyle="none")
ax[0].plot(timeStarts, radarDataBR, 'go', label="Radar", alpha=.75, markersize=3, fillstyle="none")

ax[0].set_ylabel('Breathing rate [1/min]')
ax[0].set_ylim([5,25])

ax[0].legend(loc="upper right")
ax[0].grid(visible=True, which='major', axis='y')

ax[1].plot(timeStarts, psgDataBR - radarDataBR,'o',label=f"PSG - Radar")
ax[1].plot([timeStarts[0],timeStarts[-1]+timeWinLen],[0,0],'k--',label="Diff = 0")
ax[1].set_title(f"Difference of Computed Breathing Rates (MAPE = {mape:.2f} %)")
ax[1].set_ylabel('Breathing rate difference [1/min]')

ax[1].set_ylim([-5, 5])
outOfBoundsTop = np.where((psgDataBR - radarDataBR) > 5)[0]
if any(outOfBoundsTop):
    ax[1].plot(timeStarts[outOfBoundsTop],np.zeros(len(outOfBoundsTop))+4.7,'r^',label="Diff > 5")

outOfBoundsBot = np.where((psgDataBR - radarDataBR) < -5)[0]
if any(outOfBoundsBot):
    ax[1].plot(timeStarts[outOfBoundsBot],np.zeros(len(outOfBoundsBot))-4.7,'rv',label="Diff < -5")

ax[1].legend(loc="upper right")
ax[1].set_yticks(np.arange(-4,5))
ax[1].grid(visible=True, which='major', axis='y')

cax = ax[2].matshow(bestBinsBR, extent=[timeStarts[0]-timeWinLen//2, timeStarts[-1]+timeWinLen//2, bins2check[0]*.05-.025, bins2check[-1]*.05+.025], aspect='auto', origin='lower', cmap='plasma')
ax[2].xaxis.set_ticks_position('bottom')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel("Distance from radar [m]")
ax[2].set_title("Radar Breathing Range Profile")

makeSecTax2Hour(timeStarts,ax[2])

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig6.jpg',dpi=1200,bbox_inches="tight")

plt.show()

plt.rc('font',size=10)

####################################################################################################

### Figure 7
# loading the save files for the figures
with open(path2saveFiles + f"subj10_HR_chirpsall_procData.pkl", 'rb') as fp:
    resultsDict = pickle.load(fp)

bestBinsHR =    resultsDict['selectedBins']
timeStarts =    resultsDict['timeStarts']
timeWinLen =    resultsDict['epochLen']
bins2check =    resultsDict['bins2check']
psgDataHR  =    resultsDict['psgVitalRates']
radarDataHR =   resultsDict['medianRadarVitalRates']

winsWithBoth = len(np.where((~np.isnan(radarDataHR)) & (~np.isnan(psgDataHR)))[0])
mape = (1/winsWithBoth) * np.nansum(np.abs(psgDataHR - radarDataHR) / np.abs(psgDataHR)) * 100
# end of loading the save files

plt.rc('font',size=12)
fig,ax = plt.subplots(3,1,sharex=True,figsize=(16,10))
fig.tight_layout()


ax[0].set_title("Heart Rate Computed from Radar and PSG")

ax[0].plot(timeStarts, psgDataHR, 'rs', label='PSG', alpha=.75, markersize=3, fillstyle="none")
ax[0].plot(timeStarts, radarDataHR, 'go', label="Radar", alpha=.75, markersize=3, fillstyle="none")

ax[0].set_ylabel('Heart rate [1/min]')
ax[0].set_ylim([40,120])

ax[0].legend(loc="upper right")
ax[0].grid(visible=True, which='major', axis='y')

ax[1].plot(timeStarts, psgDataHR - radarDataHR,'o',label=f"PSG - Radar")
ax[1].plot([timeStarts[0],timeStarts[-1]+timeWinLen],[0,0],'k--',label="Diff = 0")

ax[1].set_title(f"Difference of Computed Heart Rates (MAPE = {mape:.2f} %)")
ax[1].set_ylabel('Heart rate difference [1/min]')

ax[1].set_ylim([-5, 5])
outOfBoundsTop = np.where((psgDataHR - radarDataHR) > 5)[0]
if any(outOfBoundsTop):
    ax[1].plot(timeStarts[outOfBoundsTop],np.zeros(len(outOfBoundsTop))+4.7,'r^',label="Diff > 5")

outOfBoundsBot = np.where((psgDataHR - radarDataHR) < -5)[0]
if any(outOfBoundsBot):
    ax[1].plot(timeStarts[outOfBoundsBot],np.zeros(len(outOfBoundsBot))-4.7,'rv',label="Diff < -5")

ax[1].legend(loc="upper right")
ax[1].set_yticks(np.arange(-4,5))
ax[1].grid(visible=True, which='major', axis='y')

cax = ax[2].matshow(bestBinsHR, extent=[timeStarts[0]-timeWinLen//2, timeStarts[-1]+timeWinLen//2, bins2check[0]*.05-.025, bins2check[-1]*.05+.025], aspect='auto', origin='lower', cmap='plasma')
ax[2].xaxis.set_ticks_position('bottom')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel("Distance from radar [m]")
ax[2].set_title("Radar Heartbeat Range Profile")

makeSecTax2Hour(timeStarts,ax[2])

plt.tight_layout()

plt.savefig(path2saveFigs + f'fig7.jpg',dpi=1200,bbox_inches="tight")

plt.show()

plt.rc('font',size=10)

####################################################################################################

### Figure 8 - breathing Bland-Altman plot
# collecting the data from the saved files for the figure
recIDList = ["06","07","08","09","10","11"]
allResultsDict = {}
for recID in recIDList:
    
    saveFname = f"subj{recID}_BR_chirpsall_procData.pkl"
    
    analysisResSaveFile = path2saveFiles + saveFname

    with open(analysisResSaveFile, 'rb') as fp:
        resultsDict = pickle.load(fp)

    allResultsDict[recID] = resultsDict

fullPsgData   = np.array([])
fullRadarData = np.array([])

for rInd,recID in enumerate(recIDList):
    resultsDict = allResultsDict[recID]

    radarData = resultsDict['medianRadarVitalRates']
    psgData   = resultsDict['psgVitalRates']

    radarData[radarData == 0] = np.nan
    psgData[psgData == 0]     = np.nan

    cutStartInd = np.argmin(np.abs(5000 - timeStarts)) + 1
    cutEndInd = np.argmin(np.abs(25000 - timeStarts)) + 1
    radarData = radarData[cutStartInd:cutEndInd]
    psgData = psgData[cutStartInd:cutEndInd]

    fullPsgDataBR   = np.concatenate((fullPsgData,psgData))
    fullRadarDataBR = np.concatenate((fullRadarData,radarData))
# end of data loading for figure

plt.rc('font',size=12)
plt.figure(figsize=(10,6))

bothNotNan = np.where(~np.isnan(fullPsgDataBR) & ~np.isnan(fullRadarDataBR))[0]
sensorDiffsBR = fullPsgDataBR[bothNotNan] - fullRadarDataBR[bothNotNan]
sensorAvgsBR = np.mean(np.concatenate((fullPsgDataBR[bothNotNan].reshape(1,-1), fullRadarDataBR[bothNotNan].reshape(1,-1))), axis=0)

xmin = np.min(sensorAvgsBR)-3
xmax = np.max(sensorAvgsBR)+6

meanLine = np.mean(sensorDiffsBR)
upperCIline = np.mean(sensorDiffsBR) + 1.96*np.std(sensorDiffsBR, ddof=1)
lowerCIline = np.mean(sensorDiffsBR) - 1.96*np.std(sensorDiffsBR, ddof=1)

ymin = lowerCIline-.4
ymax = upperCIline+.4

plt.plot(sensorAvgsBR, sensorDiffsBR, 'bo', markersize=3, alpha=.5)

plt.hlines(y=meanLine, xmin=xmin, xmax=xmax, colors='k', linestyles='-')
plt.text(.95*xmax, meanLine+.1, f'mean={meanLine: .2f}',
         horizontalalignment='center', verticalalignment='center',
         fontsize=12, weight="normal", color='black')

plt.hlines(y=upperCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--')
plt.text(.95*xmax, upperCIline+.1, f'+1.96SD={upperCIline: .2f}',
         horizontalalignment='center', verticalalignment='center',
         fontsize=12, weight="normal", color='black')

plt.hlines(y=lowerCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--')
plt.text(.95*xmax, lowerCIline-.1, f'-1.96SD={lowerCIline: .2f}',
         horizontalalignment='center', verticalalignment='center',
         fontsize=12, weight="normal", color='black')

plt.title("Bland-Altman Plot: PSG vs Radar Breathing Rates")

plt.xlabel("Mean of sensors [BPM]")
plt.ylabel("Difference of sensors [BPM]")

plt.ylim([ymin,ymax])

plt.savefig(path2saveFigs + f'fig8.jpg',dpi=1200,bbox_inches="tight")

plt.show()
plt.rc('font',size=10)


####################################################################################################

### Figure 9
# collecting the data from the saved files for the figure
recIDList = ["06","07","08","09","10","11"]
allResultsDict = {}
for recID in recIDList:
    
    saveFname = f"subj{recID}_HR_chirpsall_procData.pkl"
    
    analysisResSaveFile = path2saveFiles + saveFname

    with open(analysisResSaveFile, 'rb') as fp:
        resultsDict = pickle.load(fp)

    allResultsDict[recID] = resultsDict

fullPsgData   = np.array([])
fullRadarData = np.array([])

for rInd,recID in enumerate(recIDList):
    resultsDict = allResultsDict[recID]

    radarData = resultsDict['medianRadarVitalRates']
    psgData   = resultsDict['psgVitalRates']

    radarData[radarData == 0] = np.nan
    psgData[psgData == 0]     = np.nan

    cutStartInd = np.argmin(np.abs(5000 - timeStarts)) + 1
    cutEndInd = np.argmin(np.abs(25000 - timeStarts)) + 1
    radarData = radarData[cutStartInd:cutEndInd]
    psgData = psgData[cutStartInd:cutEndInd]

    fullPsgDataHR   = np.concatenate((fullPsgData,psgData))
    fullRadarDataHR = np.concatenate((fullRadarData,radarData))
# end of data loading for figure
plt.rc('font',size=12)
plt.figure(figsize=(10,6))

bothNotNan = np.where(~np.isnan(fullPsgDataHR) & ~np.isnan(fullRadarDataHR))[0]
sensorDiffsHR = fullPsgDataHR[bothNotNan] - fullRadarDataHR[bothNotNan]
sensorAvgsHR = np.mean(np.concatenate((fullPsgDataHR[bothNotNan].reshape(1,-1), fullRadarDataHR[bothNotNan].reshape(1,-1))), axis=0)

xmin = np.min(sensorAvgsHR)-3
xmax = np.max(sensorAvgsHR)+6

meanLine = np.mean(sensorDiffsHR)
upperCIline = np.mean(sensorDiffsHR) + 1.96*np.std(sensorDiffsHR, ddof=1)
lowerCIline = np.mean(sensorDiffsHR) - 1.96*np.std(sensorDiffsHR, ddof=1)

ymin = lowerCIline-.4
ymax = upperCIline+.4

plt.plot(sensorAvgsHR, sensorDiffsHR, 'bo', markersize=3, alpha=.5)

plt.hlines(y=meanLine, xmin=xmin, xmax=xmax, colors='k', linestyles='-')
plt.text(.95*xmax, meanLine+.1, f'mean={meanLine: .2f}',
         horizontalalignment='center', verticalalignment='center',
         fontsize=12, weight="normal", color='black')

plt.hlines(y=upperCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--')
plt.text(.95*xmax, upperCIline+.1, f'+1.96SD={upperCIline: .2f}',
         horizontalalignment='center', verticalalignment='center',
         fontsize=12, weight="normal", color='black')

plt.hlines(y=lowerCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--')
plt.text(.95*xmax, lowerCIline-.1, f'-1.96SD={lowerCIline: .2f}',
         horizontalalignment='center', verticalalignment='center',
         fontsize=12, weight="normal", color='black')

plt.title("Bland-Altman Plot: PSG vs Radar Heart Rates")

plt.xlabel("Mean of sensors [BPM]")
plt.ylabel("Difference of sensors [BPM]")

plt.ylim([ymin,ymax])

plt.savefig(path2saveFigs + f'fig9.jpg',dpi=1200,bbox_inches="tight")

plt.show()
plt.rc('font',size=10)
####################################################################################################