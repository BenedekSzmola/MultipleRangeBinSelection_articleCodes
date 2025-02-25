########################################################################
###### import python packages ##########################################
########################################################################
import numpy as np
from scipy import signal
import copy
import pywt
########################################################################

########################################################################
### import methods from other scripts ##################################
########################################################################
import helperFunctions as uS
from persHomAlgs import subLevelFiltPersDgmChecker_DBSCAN_HR
########################################################################


########################################################################
### Start of main function #############################################
########################################################################

def checkBins(srate, data, bins2checkFull):
    ### Initialize output dictionary
    outputDict = {}

    bins2check = copy.deepcopy(bins2checkFull)

    b,a = signal.butter(N=2, Wn=[.65, 5], fs=srate, btype="bandpass")
    data2use = signal.filtfilt(b,a,data)
    
    k = np.round(.15*srate).astype("int")
    data2use = signal.filtfilt(np.ones(k)/k,1,data2use)
    
    data2use = uS.normDataTo_0_1(data2use)

    binCats   = np.zeros(len(bins2check))

    for bini,binNum in enumerate(bins2check):
        subLvlReturDict     = subLevelFiltPersDgmChecker_DBSCAN_HR(data2use[binNum,:],int(data.shape[1]/srate),showLogs=False)

        binCats[bini]   = subLvlReturDict['class']


    fullBinCats     = np.zeros(len(bins2checkFull))
    fullBinCats[np.isin(bins2checkFull,bins2check)]     = binCats
    
    outputDict = handleDetAlgIO(srate,data[bins2check,:],bins2checkFull,bins2check,values4rank=np.zeros(len(bins2check)))
    outputDict['binClasses']    = fullBinCats
        
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
        outputDict["bestBin"]  = np.full(len(bins2checkFull),np.nan)
        outputDict["heartRate"] = np.nan

        return outputDict
    
    elif numBins <= n2choose:
        chosenBinArrLocs = np.arange(len(bins2check))

    else:
        chosenBinArrLocs = np.argpartition(values4rank,-n2choose)[-n2choose:]

    chosenBins = bins2check[chosenBinArrLocs]

    numChosenBins = len(chosenBins)
    
    data2use = data4det[chosenBinArrLocs,:]

    critVals = np.full(numChosenBins, np.nan)

    heartRates = np.full(numChosenBins, np.nan)
    for bini in range(numChosenBins):

        returDict = computeHR_subLvlFilt(srate, data2use[bini,:])

        critVals[bini] = returDict['critVal']
        heartRates[bini] = returDict['heartRate']
    
    if np.all(np.isnan(heartRates)) or np.all(np.isnan(critVals)):
        outputDict["bestBin"]  = np.full(len(bins2checkFull),np.nan)
        outputDict["heartRate"] = np.nan

    else:
        if n2choose == len(bins2check):
            bestBin = np.isin(bins2checkFull,bins2check[~np.isnan(heartRates)]).astype("int")
        else:
            bestBin = (chosenBins[np.nanargmax(critVals)] == bins2checkFull).astype("int")

            if len(bestBin) == 0:
                bestBin = np.full(len(bins2check),np.nan)
        
        heartRate = np.nanmedian(heartRates)

        outputDict["bestBin"]   = bestBin
        outputDict["heartRate"] = heartRate

        allBinRates = np.full(len(bins2checkFull),np.nan)
        allBinRates[np.isin(bins2checkFull,chosenBins)] = heartRates
        outputDict['allHeartRates'] = allBinRates

    return outputDict

##################################################################################################
##################################################################################################


##################################################################################################
### The heart rate detection algorithms ######################################################
##################################################################################################


#############################################################################################################
### method subLvlFilt: peak detection based on persistance diagram that was computed via sublevel filtration
#############################################################################################################
def computeHR_subLvlFilt(srate, data):
    returDict = {'critVal':     np.nan,
                 'heartRate':  np.nan}
    
    b,a = signal.butter(N=2, Wn=[.65, 5], fs=srate, btype="bandpass")
    data2use = signal.filtfilt(b,a,data)

    k = np.round(.15*srate).astype("int")
    data2use = signal.filtfilt(np.ones(k)/k,1,data2use)

    data2use = uS.normDataTo_0_1(data2use)

    subLvlReturDict = subLevelFiltPersDgmChecker_DBSCAN_HR(data2use, int(len(data)/srate), returnPoints=True)
    lowClustBirth = subLvlReturDict['lowClustBirth']
    upperInds = subLvlReturDict['upperPOItimeinds']
    lowerInds = subLvlReturDict['lowerPOItimeinds']

    if ((lowClustBirth == "low") and (len(upperInds) < 2)) or ((lowClustBirth == "high") and (len(lowerInds) < 2)) or ((lowClustBirth in ["mixed","none"]) and ((len(lowerInds) < 2) and (len(upperInds < 2)))):
        return returDict
    
    # Helpful functions for checking whether a peak is inside the ranges
    goodFreqRange = [.65, 2]
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

    medianPeakTimeDiff = np.median(peakTimeDiffs)
    if inGoodFreqRange(1/medianPeakTimeDiff):
        if len(peakTimeDiffs) < 2:
            returDict['critVal'] = 0
        else:
            returDict['critVal'] = np.std(peakTimeDiffs, ddof=1)
        
        returDict['heartRate'] = 60 / np.median(medianPeakTimeDiff)

    return returDict

#########################################################################
### method swt: uses swt
#########################################################################
def computeHR_swt(srate, data):
    returDict = {'critVal': np.nan, 'heartRate': np.nan}

    data4det = copy.deepcopy(data)

    ###########################################
    ### compute swt
    ###########################################
    
    swtLvl = np.ceil(np.log2((srate/2) / 3.5))
    if len(data4det) % (2**swtLvl) != 0:
        necessaryLen = (2**swtLvl) * np.ceil(len(data4det) / (2**swtLvl))
        startPad = int(np.floor((necessaryLen-len(data4det))/2))
        endPad = int(np.ceil((necessaryLen-len(data4det))/2))
        dataPad = np.pad(data4det, (startPad, endPad))

    else:
        dataPad = copy.deepcopy(data4det)
        startPad = 0
        endPad = 0

    coeffs = pywt.swt(dataPad,wavelet="sym4",level=swtLvl,trim_approx=True)

    #############################
    ### Extract and square the wanted coefficient
    #############################

    COI = coeffs[1] # coeff of interest
    COI = COI[startPad:len(COI)-endPad]
    dataSquare = np.abs(signal.hilbert(COI*-1))
    
    k = np.round(srate*.3).astype("int")
    dataSquare = signal.filtfilt(np.ones(k)/k,1,dataSquare)

    #############################
    ### Compute peak2peak intervals
    #############################

    peakInds = signal.find_peaks(dataSquare, prominence=1*np.std(dataSquare,ddof=1), distance=srate/3.5)[0]
    
    if any(peakInds) and (len(peakInds) >= 3):
        peakIntervals = np.diff(peakInds) / srate

        outUpThr  = np.mean(peakIntervals) + 1*np.std(peakIntervals,ddof=1)
        outLowThr = np.mean(peakIntervals) - 1*np.std(peakIntervals,ddof=1)
        outlierInds = np.where((peakIntervals > outUpThr) | (peakIntervals < outLowThr))[0]
        outlierIvs = outlierInds[uS.intervalExtractor(outlierInds)[0]]

        for outi in range(outlierIvs.shape[0]):
            starti = outlierIvs[outi,0]
            endi = outlierIvs[outi,1]

            if starti == 0:
                preWin = np.array([])

            else:
                preWin = peakIntervals[np.max((0,starti-2)):starti]

            if endi == (len(peakIntervals)-1):
                postWin = np.array([])

            else:
                postWin = peakIntervals[endi:np.min((len(peakIntervals),endi+3))]

            surrWin = np.concatenate((preWin,postWin))
            if len(surrWin) > 0:
                peakIntervals[starti:endi+1] = np.median(surrWin)

        #########################################
        ### Test and heart activity
        #########################################
        medPeakInterval = np.mean(peakIntervals)

        std2medianRatio = np.std(peakIntervals, ddof=1) / medPeakInterval
        badIntervalsCount = np.sum( (peakIntervals < (1/3.5)) | (peakIntervals > (1/.5)) )

        
        if (std2medianRatio < .1) & (badIntervalsCount/len(peakIntervals) < .1) & (medPeakInterval >= (1/3.5)) & (medPeakInterval <= (1/.5)):
            critVal = 1/np.std(peakIntervals,ddof=1)
            heartRate = 60 / medPeakInterval # transform to /min

            returDict["critVal"] = critVal
            returDict["heartRate"] = heartRate

    return returDict

#########################################################################
### method normPeaks: simply find the peaks in the normed data
#########################################################################
def computeHR_normPeaks(srate, data):
    returDict = {'critVal': np.nan, 'heartRate': np.nan}

    b,a = signal.butter(N=1, Wn=[.8,1.7], btype="bandpass", fs=srate)
    dataFilt = signal.filtfilt(b,a,data)
    
    amps = np.abs(dataFilt + 1j*signal.hilbert(dataFilt))
    data4det = dataFilt / amps


    peakInds = signal.find_peaks(data4det, height=.9, distance=srate*60/150)[0]
    
    if not( (not any(peakInds)) or (len(peakInds) < 5) or ( np.std( np.diff(peakInds), ddof=1 )/srate > .3 ) ):
        peakIntervals = np.diff(peakInds) / srate
       
        outUpThr  = np.mean(peakIntervals) + 1*np.std(peakIntervals,ddof=1)
        outLowThr = np.mean(peakIntervals) - 1*np.std(peakIntervals,ddof=1)
        outlierInds = np.where((peakIntervals > outUpThr) | (peakIntervals < outLowThr))[0]

        if len(outlierInds)/len(peakIntervals) <= .25:
            outlierIvs = outlierInds[uS.intervalExtractor(outlierInds)[0]]

            for outi in range(outlierIvs.shape[0]):
                starti = outlierIvs[outi,0]
                endi = outlierIvs[outi,1]

                if starti == 0:
                    preWin = np.array([])

                else:
                    preWin = peakIntervals[np.max((0,starti-2)):starti]

                if endi == (len(peakIntervals)-1):
                    postWin = np.array([])

                else:
                    postWin = peakIntervals[endi:np.min((len(peakIntervals),endi+3))]

                surrWin = np.concatenate((preWin,postWin))
                if len(surrWin) > 0:
                    peakIntervals[starti:endi+1] = np.median(surrWin)
            
            #########################################
            ### Test for heart activity
            #########################################

            avgPeakInterval = np.mean(peakIntervals)

            IBIstd = np.std(peakIntervals, ddof=1)
            badIntervalsCount = np.sum( (peakIntervals < (60/150)) | (peakIntervals > (60/40)) )

            if (IBIstd < .1) & (badIntervalsCount/len(peakIntervals) < .1) & (avgPeakInterval >= (60/150)) & (avgPeakInterval <= (60/40)):

                critVal   = 1 / np.std(peakIntervals,ddof=1)
                heartRate = 60 / avgPeakInterval # transform to /min

                returDict["critVal"]   = critVal
                returDict["heartRate"] = heartRate

    return returDict
