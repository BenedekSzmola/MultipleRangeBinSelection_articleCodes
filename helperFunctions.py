import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import copy
import platform
from ripser import ripser
from persim import plot_diagrams


##############################################
### utility for figure xticks (sec --> HHMMSS)
##############################################
def convAxSec2HHMMSS(ax2conv):
    timeStampS = [item.get_text() for item in ax2conv.get_xticklabels()]    

    newStamps = ['']*len(timeStampS)
    for i,stampi in enumerate(timeStampS):
        if isinstance(stampi, str):
            stampi = int(stampi)
        stampHours = stampi // 3600
        minRem = stampi % 3600
        stampMins = minRem // 60
        stampSecs = minRem % 60
        newStamps[i] = f"{stampHours:02}:{stampMins:02}:{stampSecs:02}"

    ax2conv.set_xticklabels(newStamps, rotation=0, ha="center")

    return 
##############################################
##############################################

##############################################
### utility for sensor index
##############################################
def getSensorIdx(sensorList,sensorName):
    sensorIdx = np.where(sensorName == sensorList)[0][0]

    return sensorIdx

##############################################
##############################################

#######################################
### norm data to 0-1
#######################################
def normDataTo_0_1(data2norm):
    if len(data2norm.shape) == 1:
        if (np.nanmax(data2norm) - np.nanmin(data2norm)) == 0:
            normedData = copy.deepcopy(data2norm)
        else:
            normedData = (data2norm - np.nanmin(data2norm)) / (np.nanmax(data2norm) - np.nanmin(data2norm))
    elif len(data2norm.shape) == 2:
        normedData = (data2norm - np.nanmin(data2norm, axis=1, keepdims=True)) / (np.nanmax(data2norm, axis=1, keepdims=True) - np.nanmin(data2norm, axis=1, keepdims=True))


    return normedData
##############################################
##############################################

#######################################
###
#######################################
def intervalExtractor(x, indexCorr=False):
    # expects indices, like the output of np.where(...)[0]

    if len(x) == 0:
        return [], []

    edges = np.where( np.diff(x) != 1 )[0]
    edges = np.concatenate(([0], edges, [len(x)-1]))

    startEndInds = np.array( [edges[:-1], edges[1:]] ).T
    startEndInds[1:,0] = startEndInds[1:,0] + 1

    intervalLens = np.squeeze(np.diff(startEndInds, axis=1), axis=1) + 1

    if indexCorr:
        startEndInds[:,1] = startEndInds[:,1] + 1

    return startEndInds, intervalLens


########################################################################
### Construct sparse matrix for sublevel filtration
########################################################################
def doSubLVLfiltration(x,delInf=True,smallLifeThr=1e-3,showPlot=False,timeVec=None):
    N = len(x)

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

    if delInf:
        dgm0 = dgm0[:-1,:]
        
    dgm0 = dgm0[dgm0[:, 1]-dgm0[:, 0] > smallLifeThr, :]

    if showPlot and (timeVec is not None):
        allgrid = np.unique(dgm0.flatten())
        allgrid = allgrid[allgrid < np.inf]
        xs = np.unique(dgm0[:, 0])
        ys = np.unique(dgm0[:, 1])
        ys = ys[ys < np.inf]

        #Plot the time series and the persistence diagram
        plt.figure(figsize=(16, 6))
        ylims = [np.min(x)-.1, np.max(x)+.1]
        plt.subplot(121)
        plt.plot(timeVec, x)
        ax = plt.gca()
        ax.set_yticks(allgrid)
        ax.set_xticks([])
        plt.ylim(ylims)
        plt.grid(linewidth=1, linestyle='--')
        plt.title("Time domain signal")
        plt.xlabel("Time [s]")

        plt.subplot(122)
        ax = plt.gca()
        ax.set_yticks(ys)
        ax.set_xticks(xs)
        plt.grid(linewidth=1, linestyle='--')
        plot_diagrams(dgm0, size=50)
        plt.ylim(ylims)
        plt.title("Persistence Diagram")


        plt.show()

    return dgm0


########################################################################
### Construct Time delay embedded vector
########################################################################
def createDelayVector(x,tau,m):
    delayVec = np.zeros((len(x)-(m-1)*tau,m))

    for dimi in range(m):
        if dimi < (m-1):
            delayVec[:,dimi] = x[dimi*tau:-(m-1-dimi)*tau]
        else:
            delayVec[:,dimi] = x[dimi*tau:]

    return delayVec
########################################################################
########################################################################