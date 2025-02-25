########################################################################
###### import python packages ##########################################
########################################################################
import numpy as np
from scipy import signal
import copy

from ripser import ripser

########################################################################

########################################################################
### import methods from other scripts ##################################
########################################################################
import helperFunctions as uS
from persHomAlgs import persDiagramChecker_DBSCAN,subLevelFiltPersDgmChecker_DBSCAN
########################################################################

########################################################################
### Start of main function #############################################
########################################################################
def checkBins(srate, data, bins2checkFull):
    ### Initialize output dictionary
    outputDict = {}

    bins2check = copy.deepcopy(bins2checkFull)

    ############################################################################
    ### bin selection method
    ############################################################################

    tau = 10
    m = 3

    b,a = signal.butter(N=4, Wn=5, fs=srate, btype="lowpass")
    data2use = signal.filtfilt(b,a,data)
    
    data2use = uS.normDataTo_0_1(data2use)

    binCats   = np.zeros(len(bins2check))
    binScores = np.zeros(len(bins2check))

    for bini,binNum in enumerate(bins2check):
        embed = uS.createDelayVector(data2use[binNum,:],tau,m)

        diagrams = ripser(embed)['dgms']
        diagrams[0] = diagrams[0][:-1,:]
        TDEdiagCategory,TDEdiagScore = persDiagramChecker_DBSCAN(diagrams,int(data.shape[1]/srate),showLogs=False)
        
        subLvlReturDict     = subLevelFiltPersDgmChecker_DBSCAN(data2use[binNum,:],int(data.shape[1]/srate),showLogs=False)
        SubLVLdiagCategory  = subLvlReturDict['class']

        if (TDEdiagCategory == 2) or (SubLVLdiagCategory == 2):
            diagCategory = 2
        else:
            diagCategory = np.min((TDEdiagCategory,SubLVLdiagCategory))

        diagScore = copy.deepcopy(TDEdiagScore)
        
        binCats[bini]   = diagCategory
        binScores[bini] = diagScore


    fullBinCats     = np.zeros(len(bins2checkFull))
    fullBinScores   = np.zeros(len(bins2checkFull))
    fullBinCats[np.isin(bins2checkFull,bins2check)]     = binCats
    fullBinScores[np.isin(bins2checkFull,bins2check)]   = binScores
    
    goodBinInds = np.isin(binCats, [1,2])
    binScores = binScores[goodBinInds]
    goodBins2check = bins2check[goodBinInds]

    if len(goodBins2check) == 0:
        outputDict["bestBin"] = np.full(len(bins2checkFull),np.nan)
        outputDict["respRate"] = np.nan

        outputDict['binClasses'] = fullBinCats
        outputDict['binScores']  = fullBinScores

        return outputDict

    outputDict = handleDetAlgIO(srate,data[goodBins2check,:],bins2checkFull,goodBins2check,values4rank=binScores)

    outputDict['binClasses']    = fullBinCats
    outputDict['binScores']     = fullBinScores

    return outputDict

##############################################################################################################################################################################
##############################################################################################################################################################################

##############################################################################################################################################################################
### A helper function which handles the bins which were selected by the bin selection algorithms, and the computed vital rates
##############################################################################################################################################################################
def handleDetAlgIO(srate, data4det, bins2checkFull, bins2check, values4rank):
    '''
    Input parameters:
    -----------------
    srate:              sampling rate of input data
    data4det:           the data which should be used for the further processing steps (expected in bins x samples format)
    bins2checkFull:     the ID of all the bins which were selected when starting the detection
    bins2check:         the ID number of the bins which are still under consideration after the previous steps
    values4rank:        the values by which the bins can be ranked
    ====================================================================================================================================================

    Returns:
    --------
    outputDict:         a dictionary with at least 2 keys: "bestBin" and "respRate", which are the chosen range bin and the computed vital rate respectively
    ========================================================================================================================================================
    '''
    
    outputDict = {}

    numBins = len(bins2check)

    n2choose = len(bins2check)

    if numBins == 0:
        outputDict["bestBin"] = np.full(len(bins2checkFull),np.nan)
        outputDict["respRate"] = np.nan

        return outputDict
    
    elif numBins <= n2choose:
        chosenBinArrLocs = np.arange(len(bins2check))

    else:
        chosenBinArrLocs = np.argpartition(values4rank,-n2choose)[-n2choose:]

    chosenBins = bins2check[chosenBinArrLocs]

    numChosenBins = len(chosenBins)
    
    data2use = data4det[chosenBinArrLocs,:]

    critVals = np.full((numChosenBins,1), np.nan)
    breathRates = np.full(numChosenBins, np.nan)
    for bini in range(numChosenBins):
        returDict = computeBR_subLvlFilt(srate, data2use[bini,:])

        critVals[bini,:]  = returDict['critVal']
        breathRates[bini] = returDict['breathRate']
    
    critVals = critVals[:,0]

    if np.all(np.isnan(breathRates)) or np.all(np.isnan(critVals)):
        outputDict["bestBin"] = np.full(len(bins2checkFull),np.nan)
        outputDict["respRate"] = np.nan

    else:
        
        if n2choose == len(bins2check):
            bestBin = np.isin(bins2checkFull,bins2check[~np.isnan(breathRates)]).astype("int")
        else:
            bestBin = (chosenBins[np.nanargmax(critVals)] == bins2checkFull).astype("int")

        breathRate = np.nanmedian(breathRates)
            
        outputDict["bestBin"]  = bestBin
        outputDict["respRate"] = breathRate

        allBinRates = np.full(len(bins2checkFull),np.nan)
        allBinRates[np.isin(bins2checkFull,chosenBins)] = breathRates
        outputDict['allRespRates'] = allBinRates

    return outputDict

##################################################################################################
##################################################################################################


##################################################################################################
### The breathing rate detection algorithms ######################################################
##################################################################################################

def computeBR_subLvlFilt(srate, data):
    returDict = {'critVal':     np.nan,
                 'breathRate':  np.nan}
    
    b,a = signal.butter(N=4, Wn=5, fs=srate, btype="lowpass")
    data2use = signal.filtfilt(b,a,data)
    
    data2use = uS.normDataTo_0_1(data2use)

    subLvlReturDict = subLevelFiltPersDgmChecker_DBSCAN(data2use, int(len(data)/srate), returnPoints=True, showLogs=False)
    lowClustBirth = subLvlReturDict['lowClustBirth']
    upperInds = subLvlReturDict['upperPOItimeinds']
    lowerInds = subLvlReturDict['lowerPOItimeinds']

    if ((lowClustBirth == "low") and (len(upperInds) < 2)) or ((lowClustBirth == "high") and (len(lowerInds) < 2)) or ((lowClustBirth in ["mixed","none"]) and ((len(lowerInds) < 2) and (len(upperInds < 2)))):
    
        return returDict
    
    # Helpful functions for checking whether a peak is inside the ranges
    goodFreqRange = [.15, .4]
    inGoodFreqRange = lambda f: (f >= goodFreqRange[0]) & (f <= goodFreqRange[1])
    
    if lowClustBirth in ["mixed","none"]:
        upperIndDiffs = np.diff(upperInds)
        lowerIndDiffs = np.diff(lowerInds)
        
        upperPeakTimeDiffs = upperIndDiffs / srate
        lowerPeakTimeDiffs = lowerIndDiffs / srate

        peakTimeDiffs = np.hstack((upperPeakTimeDiffs,lowerPeakTimeDiffs))
    elif lowClustBirth in ["low","high"]:
        if lowClustBirth == "low":
            peakIndDiffs = np.diff(upperInds)
        elif lowClustBirth == "high":
            peakIndDiffs = np.diff(lowerInds)

        peakTimeDiffs = peakIndDiffs / srate
    else:
        return returDict
    
    # filter peak diffs individually
    peakTimeDiffs = peakTimeDiffs[inGoodFreqRange(1/peakTimeDiffs)]
    if len(peakTimeDiffs) == 0:
        return returDict

    meanPeakTimeDiff = np.mean(peakTimeDiffs)
    if inGoodFreqRange(1/meanPeakTimeDiff):
        if len(peakTimeDiffs) < 2:
            returDict['critVal'] = 0
        else:
            returDict['critVal'] = np.std(peakTimeDiffs, ddof=1)
        
        returDict['breathRate'] = 60 / meanPeakTimeDiff

    return returDict

#########################################################################
### method acorr
#########################################################################

def computeBR_acorr(srate, data):

    b,a = signal.butter(N=2,Wn=[.1,.4], btype='bandpass', fs=srate)
    dataFilt = signal.filtfilt(b,a,data)

    ##############################
    ### Compute autocorrelation
    ##############################

    autocorr = signal.correlate(dataFilt, dataFilt)
    autocorr = autocorr / np.max(autocorr)

    autocorr_lags = signal.correlation_lags(len(dataFilt), len(dataFilt))
    autocorr_lags = autocorr_lags / srate

    firstNegPeaks = signal.find_peaks(-autocorr)[0]
    if len(firstNegPeaks) < 2:
        return {'critVal': np.full(2, np.nan), 'breathRate': np.nan}
    else:
        firstNegPeaks = firstNegPeaks[np.argpartition(np.abs(autocorr_lags[firstNegPeaks]),1)[:2]]
        acorrWidth = autocorr_lags[firstNegPeaks[1]] - autocorr_lags[firstNegPeaks[0]]
        if (acorrWidth < 60/25) or (acorrWidth > 60/5):
            return {'critVal': np.full(2, np.nan), 'breathRate': np.nan}

    #########################################
    ### Test for breathing activity
    #########################################

    acorrPosInds = np.where((autocorr_lags >= 60/25) & (autocorr_lags <= np.min((2.1*60/5,np.max(autocorr_lags)))) )[0]

    acorrPeaks,_ = signal.find_peaks(autocorr[acorrPosInds],distance=np.round(3*srate).astype("int"),height=0)

    critVal    = np.full(2, np.nan)
    breathRate = np.nan

    if any(acorrPeaks) and (len(acorrPeaks) > 1):
        acorrPeaks = acorrPosInds[acorrPeaks]

        interPeakStd = np.std(np.diff(autocorr_lags[acorrPeaks],prepend=0),ddof=1)

        if (interPeakStd < 1) and (autocorr_lags[acorrPeaks[0]] >= (1/.5)) and (autocorr_lags[acorrPeaks[0]] <= (1/.1)):
            critVal    = np.array( [np.sum(autocorr[acorrPeaks]) , 1 / interPeakStd] )
            breathRate = 60 / autocorr_lags[acorrPeaks[0]]

    returDict = {'critVal': critVal,
                 'breathRate': breathRate}

    return returDict
