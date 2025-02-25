"""
Author of the scripts and documentation: Benedek Szmola

Main script to execute the analysis presented in the paper "Enhancing Vital Sign Monitoring with Laterally Placed Radar Using Persistence Diagram-Based Multiple Range Bin Selection"
Some functions for e.g. loading the data are stored in separate code files, and are called by this script.

Scripts for computing the statistics, and creating the figures (excluding Fig. 1) are contained in separate files (computeStats.py & createFigs.py respectively)


There are 3 parts of the data analysis:
- loading the raw recordings and extracting the radar and reference (PSG) signals
- running persistence homology based range bin classification in 15 second windows with 5 second steps
- take the classified 15 second windows and merge them into 60 second windows, then compute vital rates

For the publication, the same raw data was used as in the following study: 
Hornig, L.; Szmola, B.; Pätzold, W.; Vox, J.P.; Wolf, K.I. Evaluation of Lateral Radar Positioning for Vital Sign Monitoring: An Empirical Study. Sensors 2024, 24, 3548. https://doi.org/10.3390/s24113548



"""

### module imports #################################################################################################
# import python libraries
import csv
import pickle
import platform

import numpy as np
from scipy import stats

# import from other files of the project
from readRecordingData import *
import breathing_classification_rateCompute as bD
import heartbeat_classification_rateCompute as hD
import helperFunctions as uS
#####################################################################################################################

with open('radarSettings.csv') as csv_file:
    reader = csv.reader(csv_file)
    radarSettings = dict(reader)

radarSettings = {key:float(value) for key,value in radarSettings.items()}
for key,value in radarSettings.items():
    if value.is_integer():
        radarSettings[key] = int(value)

### This function does the initial bin classification in 15 second windows
def initBinClassification(recID, breathOrHeart, showLogs=False):
    ### loading in the data, some parameters for that ###################################################################
    radar_idx = 38
    filePath = f"placeholder_subj{recID}" # insert the path here for the data you want to use
    #################################################################################

    # Load the measurement from its save file
    file_info,_,synchro_info,measurement_data = readSaveFile(file_name=filePath)

    radar_srate = synchro_info['Effective sampling frequency given by xdf.load() (Radar_1, Radar_2, Radar_5)'][radar_idx-38]
    #####################################################################################################################

    ### setup for the time interval to check ############################################################################
    epochInputDict = {
        'timeStart': 0,
        'timeEnd' : np.inf,
        'epochStepSize': 5
    }

    epochLen = 15
    #####################################################################################################################

    ### Making epochs out of the full radar data, and settings for that #################################################
    # Sets whether to use Hann window before computing the range FFT
    useHannForRadarRange = True

    # Here we select which chirps are used. For the computation of 15 s bin classification, always every chirp is used
    radarChirps2use = np.arange(radarSettings['radar_loop_num'])

    # This determines that the median is taken over the 12 chirps
    chirpSumMeth = "median"

    timestampEpochs,phaseEpochs,_,epochStartTimestamps,timeStarts = readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,radar_srate,epochLen,epochInput=epochInputDict,useHann=useHannForRadarRange,chirp=radarChirps2use,chirpSumMethod=chirpSumMeth,doUnwrap=True,norm01=False,removeDC=True)
    #####################################################################################################################

    ### Selecting what type of analysis to do (BR/HR) and then loading the correct reference sensors ####################

    if breathOrHeart == "BR":
        refSensor_idx = uS.getSensorIdx(file_info['Measurement Data Format'],"Thorax")
        
    elif breathOrHeart == "HR":
        refSensor_idx = uS.getSensorIdx(file_info['Measurement Data Format'],"ECG II")

    refSensor_srate = int(float(file_info['Measurement Data Sampling Rate'][refSensor_idx]))
    refTimestampsFull,refDataFull = readFullRefData(measurement_data,refSensor_idx)
    #####################################################################################################################


    ### selecting which range bins to work with #########################################################################
    bins2check = np.arange(9,61)
    #####################################################################################################################

    ### initializing vectors for the classified bins ####################################################################
    binClasses  = np.full((len(bins2check),len(timestampEpochs)),np.nan)

    ##################################################################################################################### 

    ### starting the loop through the selected time interval ############################################################
    for epochi in range(len(timestampEpochs)):
        if showLogs:
            print('##############################################')
            print(f"### Starting epoch {epochi} | start timestamp: {epochStartTimestamps[epochi]:.4f} | seconds from recording's beginning: {timeStarts[epochi]:.0f} ")

        ### Extract radar data
        currPhase     = phaseEpochs[epochi]
            
        if breathOrHeart == "BR":
            outputDict = bD.checkBins(srate=radar_srate, data=currPhase, bins2checkFull=bins2check)
            
            binClasses[:,epochi] = outputDict['binClasses']

            if showLogs:
                print('\n#########################################')
                print(f"Epoch timestamp: {epochStartTimestamps[epochi]:.4f}, {timeStarts[epochi]:.0f} s from start")
                print('#########################################')

        elif breathOrHeart == "HR":
            outputDict = hD.checkBins(srate=radar_srate, data=currPhase, bins2checkFull=bins2check)
            
            binClasses[:,epochi] = outputDict['binClasses']

            if showLogs:
                print('\n#########################################')
                print(f"Epoch timestamp: {epochStartTimestamps[epochi]:.4f}, {timeStarts[epochi]:.0f} s from start")
                print('#########################################')

    return file_info,measurement_data,radar_srate,epochStartTimestamps,timeStarts,refSensor_idx,refTimestampsFull,refDataFull,refSensor_srate,binClasses
    #####################################################################################################################
    #####################################################################################################################


### This is the function for merging initially classified bins 
def mergeClassifiedBins(breathOrHeart, file_info, measurement_data, radar_srate, epochStartTimestamps, timeStarts, refSensor_idx, refTimestampsFull, refDataFull, refSensor_srate, binClasses, radarChirps2useInput, showLogs=False):
    radar_idx = 38
    ### Making epochs out of the full radar data, and settings for that #################################################
    useHannForRadarRange = True

    if isinstance(radarChirps2useInput, str) and (radarChirps2useInput == "all"):
        print('Using all of the chirps')
        radarChirps2use = np.arange(radarSettings['radar_loop_num'])
        computeBinStats = True
    else:
        print('Using the first chirp')
        radarChirps2use = 0
        computeBinStats = False

    # How to summarize the chirps (mean or median)
    chirpSumMeth = "median"

    #####################################################################################################################

    numContWin = 10
    numAllowedGaps = 3
    bins2check = np.arange(9,61)

    if breathOrHeart == "BR":
        detMethod = 'acorr'
        refDetMethod = 'acorr'
    elif breathOrHeart == "HR":
        detMethod = 'normPeaks'
        refDetMethod = "swt"

    #####################################################################################################################

    contiBestBins = copy.deepcopy(binClasses)

    temp = copy.deepcopy(binClasses)

    for bini in range(contiBestBins.shape[0]):
        for epochi in range(contiBestBins.shape[1]):
            if np.sum(contiBestBins[bini,epochi:epochi+numContWin] > 0) >= (numContWin-numAllowedGaps):
                temp[bini,epochi:epochi+numContWin] = 10

        temp[bini,temp[bini,:] < 10] = 0

    contiBestBins = copy.deepcopy(temp)
    #####################################################################################################################

    #
    vitalRateFromContWins = np.full((len(bins2check),len(epochStartTimestamps)),np.nan)
    vitalRateBinAvg = np.full(len(epochStartTimestamps),np.nan)
    vitalRateBinMedian = np.full(len(epochStartTimestamps),np.nan)
    vitalRateRef = np.full(len(epochStartTimestamps),np.nan)

    #
    if computeBinStats:
        selBinsMagSkew  = np.full(len(epochStartTimestamps),np.nan)
        selBinsMagKurto = np.full(len(epochStartTimestamps),np.nan)
        selBinsMagStd   = np.full(len(epochStartTimestamps),np.nan)

        badBinsMagSkew  = np.full(len(epochStartTimestamps),np.nan)
        badBinsMagKurto = np.full(len(epochStartTimestamps),np.nan)
        badBinsMagStd   = np.full(len(epochStartTimestamps),np.nan)

        quantileForStd = .9

    for epochi in range(len(epochStartTimestamps)-10):
        if showLogs:
            print('')
            print('-----------------------------------------')
            print(f'Currently on epoch {epochi}/{len(epochStartTimestamps)}')

        epochInputDict = {
            'epochStarts': np.array([epochStartTimestamps[epochi]])
        }

        _,phaseEpochs,magnitudeEpochs,currEpochStartTimestamps,_ = readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,radar_srate,60,epochInput=epochInputDict,useHann=useHannForRadarRange,chirp=radarChirps2use,chirpSumMethod=chirpSumMeth,doUnwrap=True,norm01=False,removeDC=True)
        currPhase = phaseEpochs[0]
        if computeBinStats:
            currMag = magnitudeEpochs[0]

        refEpochInputDict = {'radarEpochStarts': currEpochStartTimestamps}
        _,refEpochs,_,_ = makeRefDataEpochs(refTimestampsFull,refDataFull,refSensor_srate,60,refEpochInputDict,norm01=True)
        currRef = refEpochs[0]

        if breathOrHeart == "BR":
            refReturDict = bD.computeBR_acorr(refSensor_srate, currRef)

            vitalRateRef[epochi] = refReturDict['breathRate']
        
        elif breathOrHeart == "HR":
            refReturDict = hD.computeHR_swt(refSensor_srate, currRef)

            vitalRateRef[epochi] = refReturDict['heartRate']

        # Placeholders for computing statistics on the magnitude timeseries for comparing selected and non-selected bins
        if computeBinStats:
            currSelBinsSkew  = np.full(len(bins2check),np.nan)
            currSelBinsKurto = np.full(len(bins2check),np.nan)
            currSelBinsStd   = np.full(len(bins2check),np.nan)

            currBadBinsSkew  = np.full(len(bins2check),np.nan)
            currBadBinsKurto = np.full(len(bins2check),np.nan)
            currBadBinsStd   = np.full(len(bins2check),np.nan)

        for bini,binNum in enumerate(bins2check):
            if np.sum(binClasses[bini,epochi:epochi+numContWin] > 0) >= (numContWin-numAllowedGaps):
                if breathOrHeart == "BR":
                    returDict = bD.computeBR_acorr(radar_srate, currPhase[binNum,:])
                    
                    vitalRateFromContWins[bini,epochi] = returDict['breathRate']

                elif breathOrHeart == "HR":
                    returDict = hD.computeHR_normPeaks(radar_srate, currPhase[binNum,:])
                    
                    vitalRateFromContWins[bini,epochi] = returDict['heartRate']

                if computeBinStats:
                    currSelBinsSkew[bini]  = np.abs(stats.skew(currMag[binNum,:]))
                    currSelBinsKurto[bini] = np.abs(stats.kurtosis(currMag[binNum,:]))
                    currSelBinsStd[bini]   = np.std(currMag[binNum,:], ddof=1)
            elif computeBinStats:
                currBadBinsSkew[bini]  = np.abs(stats.skew(currMag[binNum,:]))
                currBadBinsKurto[bini] = np.abs(stats.kurtosis(currMag[binNum,:]))
                currBadBinsStd[bini]   = np.std(currMag[binNum,:], ddof=1)

        if np.any(~np.isnan(vitalRateFromContWins[:,epochi])):
            vitalRateBinAvg[epochi] = np.nanmean(vitalRateFromContWins[:,epochi])
            vitalRateBinMedian[epochi] = np.nanmedian(vitalRateFromContWins[:,epochi])

        if computeBinStats:
            selBinsMagSkew[epochi]  = np.nanmean(currSelBinsSkew)
            selBinsMagKurto[epochi] = np.nanmean(currSelBinsKurto)
            selBinsMagStd[epochi]   = np.nanquantile(currSelBinsStd, quantileForStd)

            badBinsMagSkew[epochi]  = np.nanmean(currBadBinsSkew)
            badBinsMagKurto[epochi] = np.nanmean(currBadBinsKurto)
            badBinsMagStd[epochi]   = np.nanquantile(currBadBinsStd, quantileForStd)

        ##########################################################################################
        
    #####################################################################################################################

    ### save the results as a dictionary to pkl file ####################################################################

    saveDict = {'recID': recID, 'vitalType': breathOrHeart, 'radarIdx': radar_idx, 'radarPos': "Footend", 'bins2check': bins2check,
                'numContWin': numContWin, 'numAllowedGaps': numAllowedGaps, 'epochLen': 60, 'epochStarts': epochStartTimestamps, 'timeStarts': timeStarts,
                'useHannForRadarRange': useHannForRadarRange, 'radarChirps2use': radarChirps2use, 'radarChirpSumMethod': chirpSumMeth, 
                'detMethod': detMethod, 'refSensor': file_info['Measurement Data Format'][refSensor_idx], 'refMethod': refDetMethod, 'selectedBins': contiBestBins,
                'medianRadarVitalRates': vitalRateBinMedian, 'meanRadarVitalRates': vitalRateBinAvg, 'radarAllBinVitalRates': vitalRateFromContWins, 'psgVitalRates': vitalRateRef}

    if computeBinStats:
        saveDict['selBinsMagSkew']  = selBinsMagSkew
        saveDict['selBinsMagKurto'] = selBinsMagKurto
        saveDict['selBinsMagStd']   = selBinsMagStd
        saveDict['badBinsMagSkew']  = badBinsMagSkew
        saveDict['badBinsMagKurto'] = badBinsMagKurto
        saveDict['badBinsMagStd']   = badBinsMagStd
        saveDict['quantileForStd']  = quantileForStd

    analysisResSaveFile = f"subj{recID}_{breathOrHeart}_chirps{radarChirps2useInput}_procData.pkl"

    if platform.system() == "Linux":
        analysisResSaveFile = "processedData//" + analysisResSaveFile
    elif platform.system() == "Windows":
        analysisResSaveFile = "processedData\\" + analysisResSaveFile

    with open(analysisResSaveFile, 'wb') as fp:
        pickle.dump(saveDict, fp)

    return
    #####################################################################################################################


### The functions are called here to reproduce the results from the article

# The list of subjects included in the analysis
subjList = ["06","07","08","09","10","11"]

for recID in subjList:
    print("Starting with subject",recID)
    for brORhr in ["BR", "HR"]:
        print('Doing analysis for: ',brORhr)
        file_info, measurement_data, radar_srate, epochStartTimestamps, timeStarts, refSensor_idx, refTimestampsFull, refDataFull, refSensor_srate, binClasses = initBinClassification(recID=recID, breathOrHeart=brORhr, showLogs=False)
        print('Initial bin classification ready, moving on to merging part...')
        for radarChirps2use in ["all", 0]:
            print('Using chirp(s): ',radarChirps2use)
            mergeClassifiedBins(breathOrHeart=brORhr, file_info=file_info, measurement_data=measurement_data, radar_srate=radar_srate, timeStarts=timeStarts, epochStartTimestamps=epochStartTimestamps,
                                refSensor_idx=refSensor_idx, refTimestampsFull=refTimestampsFull, refDataFull=refDataFull, refSensor_srate=refSensor_srate,
                                binClasses=binClasses, radarChirps2useInput=radarChirps2use, showLogs=False)
            print('Done with the analyses!')