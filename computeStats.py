import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import platform

import helperFunctions as uS


### Function for loading the processed data and computing the results presented in the article
def compArticleResults(breathOrHeart, radarChirps2use):

    recIDlist = ["06","07","08","09","10","11"]

    doCut = True

    if isinstance(radarChirps2use, str) and (radarChirps2use == "all"):
        computeBinStats = True
    else: 
        computeBinStats = False


    # Placeholders for collecting results (bin classification & vital rates) from each subject into this dictionary
    allResultsDict = {}

    # Placeholders for collecting the results for the magnitude based analysis on the selected/non-selected bins
    if computeBinStats:
        aggregSkew_winsWithSel  = np.array([])
        aggregKurto_winsWithSel = np.array([])
        aggregStd_winsWithSel   = np.array([])

        aggregSkew_wins0sel  = np.array([])
        aggregKurto_wins0sel = np.array([])
        aggregStd_wins0sel   = np.array([])

    for recID in recIDlist:
        
        #########################################################################
        ### 
        if platform.system() == "Linux":
            subDir = "processedData//"
        elif platform.system() == "Windows":
            subDir = "processedData\\"
        
        saveFname = f"subj{recID}_{breathOrHeart}_chirps{radarChirps2use}_procData.pkl"
        
        analysisResSaveFile = subDir + saveFname

        with open(analysisResSaveFile, 'rb') as fp:
            resultsDict = pickle.load(fp)

        allResultsDict[recID] = resultsDict

        # Collecting for the magnitude analysis
        if computeBinStats:
            selectedBins = resultsDict['selectedBins']

            quantileForStd = resultsDict['quantileForStd']

            selBinsMagSkew  = resultsDict["selBinsMagSkew"]
            selBinsMagKurto = resultsDict["selBinsMagKurto"]
            selBinsMagStd   = resultsDict["selBinsMagStd"]

            badBinsMagSkew  = resultsDict["badBinsMagSkew"]
            badBinsMagKurto = resultsDict["badBinsMagKurto"]
            badBinsMagStd   = resultsDict["badBinsMagStd"]

            winsWithNoSelBins = np.all((selectedBins == 0), axis=0)
                
            aggregSkew_winsWithSel  = np.concatenate((aggregSkew_winsWithSel,selBinsMagSkew[~winsWithNoSelBins]))
            aggregKurto_winsWithSel = np.concatenate((aggregKurto_winsWithSel,selBinsMagKurto[~winsWithNoSelBins]))
            aggregStd_winsWithSel   = np.concatenate((aggregStd_winsWithSel,selBinsMagStd[~winsWithNoSelBins]))

            aggregSkew_wins0sel  = np.concatenate((aggregSkew_wins0sel,badBinsMagSkew[winsWithNoSelBins]))
            aggregKurto_wins0sel = np.concatenate((aggregKurto_wins0sel,badBinsMagKurto[winsWithNoSelBins]))
            aggregStd_wins0sel   = np.concatenate((aggregStd_wins0sel,badBinsMagStd[winsWithNoSelBins]))
        #########################################################################

    #########################################################################
    ### Here is the computation of the magnitude timeseries based comparison between selected and non-selected bins
    if computeBinStats:
        print(f'Num windows with {breathOrHeart} selections: {len(aggregKurto_winsWithSel)}')
        print(f'Num windows without {breathOrHeart} selections: {len(aggregKurto_wins0sel)}')

        plt.figure(figsize=(8,4))
        sns.boxplot((aggregSkew_winsWithSel,aggregSkew_wins0sel))
        plt.title(f'subjects {recIDlist} {breathOrHeart} magnitude | Mean skewness of selected bins vs all bins in windows w/o selection')
        plt.xticks([0,1],["Selected bins","Wins w/o selections"])
        plt.show()

        denaned1 = copy.deepcopy(aggregSkew_winsWithSel)
        denaned1 = denaned1[~np.isnan(denaned1)]
        denaned2 = copy.deepcopy(aggregSkew_wins0sel)
        denaned2 = denaned2[~np.isnan(denaned2)]
        print(f'subjects {recIDlist} {breathOrHeart} magnitude | Ratio of mean window abs skewness > 1 when bins selected: {np.sum(np.abs(denaned1) > 1)/len(denaned1):.4f}')
        print(f'subjects {recIDlist} {breathOrHeart} magnitude | Ratio of mean window abs skewness > 1 when no bin selected: {np.sum(np.abs(denaned2) > 1)/len(denaned2):.4f}')

        #########

        plt.figure(figsize=(8,4))
        sns.boxplot((aggregKurto_winsWithSel,aggregKurto_wins0sel))
        plt.title(f'subjects {recIDlist} {breathOrHeart} magnitude | Mean kurtosis of selected bins vs all bins in windows w/o selection')
        plt.xticks([0,1],["Selected bins","Wins w/o selections"])
        plt.show()

        denaned1 = copy.deepcopy(aggregKurto_winsWithSel)
        denaned1 = denaned1[~np.isnan(denaned1)]
        denaned2 = copy.deepcopy(aggregKurto_wins0sel)
        denaned2 = denaned2[~np.isnan(denaned2)]
        print(f'subjects {recIDlist} {breathOrHeart} magnitude | Ratio of mean window abs kurtosis > 3 when bins selected: {np.sum(np.abs(denaned1) > 3)/len(denaned1):.4f}')
        print(f'subjects {recIDlist} {breathOrHeart} magnitude | Ratio of mean window abs kurtosis > 3 when no bin selected: {np.sum(np.abs(denaned2) > 3)/len(denaned2):.4f}')

        #########

        plt.figure(figsize=(8,4))
        sns.boxplot((aggregStd_winsWithSel,aggregStd_wins0sel))
        plt.title(f'subjects {recIDlist} {breathOrHeart} magnitude | {quantileForStd*100}th percentile STD of selected bins vs all bins in windows w/o selection')
        plt.xticks([0,1],["Selected bins","Wins w/o selections"])
        plt.show()

        denaned1 = copy.deepcopy(aggregStd_winsWithSel)
        denaned1 = denaned1[~np.isnan(denaned1)]
        denaned2 = copy.deepcopy(aggregStd_wins0sel)
        denaned2 = denaned2[~np.isnan(denaned2)]
        print(f'subjects {recIDlist} {breathOrHeart} magnitude | median and IQR in selected bins: {np.median(denaned1):.2f} | {np.quantile(denaned1, .25):.2f} - {np.quantile(denaned1, .75):.2f}')
        print(f'subjects {recIDlist} {breathOrHeart} magnitude | median and IQR in bins of windows w/o selections: {np.median(denaned2):.2f} | {np.quantile(denaned2, .25):.2f} - {np.quantile(denaned2, .75):.2f}')
        #####################################################################################################################



    fullPsgData   = np.array([])
    fullRadarData = np.array([])
    numWins            = 0
    timeInSec          = 0
    winsWithPSG        = 0
    winsWithRadar      = 0
    winsWithBoth       = 0
    winsWithPsgNoRadar = 0
    winsWithNoPsgRadar = 0
    winsWithNeither    = 0
    winsWithinCritDist = 0
    winsRealClose      = 0
    psgNonCoveredSec   = 0
    radarNonCoveredSec = 0
    psgAllBadIntervalLens = np.array([])
    radarAllBadIntervalLens = np.array([])

    for recID in recIDlist:
        resultsDict = allResultsDict[recID]

        timeStarts   = resultsDict['timeStarts']
        timeStep     = timeStarts[1] - timeStarts[0]
        timeWinLen   = resultsDict['epochLen']
        bestBins     = resultsDict['selectedBins']

        radarData = resultsDict['medianRadarVitalRates']
        psgData   = resultsDict['psgVitalRates']

        bestBins[:,radarData == 0]  = np.nan
        radarData[radarData == 0] = np.nan
        psgData[psgData == 0]     = np.nan

        # if you want to cut the data and also have the stats reflect that:
        if doCut:
            cutStartInd = np.argmin(np.abs(5000 - timeStarts)) + 1
            cutEndInd = np.argmin(np.abs(25000 - timeStarts)) + 1
            bestBins = bestBins[:,cutStartInd:cutEndInd]
            radarData = radarData[cutStartInd:cutEndInd]
            psgData = psgData[cutStartInd:cutEndInd]
            timeStarts = timeStarts[cutStartInd:cutEndInd]

        if np.sum(np.isnan(psgData)) > (timeWinLen/timeStep):
            _,badIntervalLens = uS.intervalExtractor(np.where(np.isnan(psgData))[0])
            badIntervalLens = badIntervalLens[badIntervalLens >= (timeWinLen/timeStep)] - (timeWinLen/timeStep - 1)
            psgAllBadIntervalLens = np.concatenate((psgAllBadIntervalLens,badIntervalLens))
            psgNonCoveredSec += np.sum(badIntervalLens) * timeStep
        else:
            psgNonCoveredSec = 0

        if np.sum(np.isnan(radarData)) > (timeWinLen/timeStep):
            _,badIntervalLens = uS.intervalExtractor(np.where(np.isnan(radarData))[0])
            badIntervalLens = badIntervalLens[badIntervalLens >= (timeWinLen/timeStep)] - (timeWinLen/timeStep - 1)
            radarAllBadIntervalLens = np.concatenate((radarAllBadIntervalLens,badIntervalLens))
            radarNonCoveredSec += np.sum(badIntervalLens) * timeStep
        else:
            radarNonCoveredSec = 0

        fullPsgData   = np.concatenate((fullPsgData,psgData))
        fullRadarData = np.concatenate((fullRadarData,radarData))
        
        # how many timewindows had detected activity PSG & radar
        numWins            += len(timeStarts)

        timeInSec          += timeStarts[-1] + timeWinLen - timeStarts[0]

        winsWithPSG        += len(np.where(~np.isnan(psgData))[0])

        winsWithRadar      += len(np.where(~np.isnan(radarData))[0])

        winsWithBoth       += len(np.where((~np.isnan(radarData)) & (~np.isnan(psgData)))[0])

        winsWithPsgNoRadar += len(np.where((np.isnan(radarData)) & (~np.isnan(psgData)))[0])

        winsWithNoPsgRadar += len(np.where((~np.isnan(radarData)) & (np.isnan(psgData)))[0])

        winsWithNeither    += len(np.where((np.isnan(radarData)) & (np.isnan(psgData)))[0])

        # how many timewindows had PSG & radar within a certain value
        if breathOrHeart == "BR":
            crtiDist = 3

        elif breathOrHeart == "HR":
            crtiDist = 5

        winsWithinCritDist += len( np.where( np.abs(psgData - radarData) <= crtiDist )[0] )

        boundOfUncertain = (1/timeWinLen) * 60
        winsRealClose += len( np.where( np.abs(psgData - radarData) <= boundOfUncertain )[0] )

    print('##################################################')
    print('Results from subdir: ')
    print(subDir)
    print('##################################################\n')

    print(f'######### {"Respiratory rate" if breathOrHeart=="BR" else "Heart rate"} analysis subjects: {recIDlist} #########')
    print('##################################################\n')

    recHours = timeInSec // 3600
    minRem = timeInSec % 3600
    recMins = minRem // 60
    recSecs = minRem % 60
    recDurStr = f"{recHours:.0f} hours, {recMins:.0f} minutes, {recSecs:.1f} seconds"
    print(f"Total number of time windows: {numWins} , with window length: {timeWinLen} s , step of {int(timeStep)} s , full duration analysed: {recDurStr}")
    print(f"Percent of windows with detected PSG {breathOrHeart} activity: {100*winsWithPSG/numWins: .1f}% ({winsWithPSG} windows)")
    print(f"Percent of windows with detected Radar {breathOrHeart} activity: {100*winsWithRadar/numWins: .1f}% ({winsWithRadar} windows)")
    print(f"Percent of windows with both PSG and Radar detected {breathOrHeart} activity: {100*winsWithBoth/numWins: .1f}% ({winsWithBoth} windows)")
    print('')

    nonCovH = psgNonCoveredSec // 3600
    minRem = psgNonCoveredSec % 3600
    nonCovMin = minRem // 60
    nonCovS = minRem % 60
    nonCovStr = f"{nonCovH:.0f} hours, {nonCovMin:.0f} minutes, {nonCovS:.1f} seconds"
    print(f'Seconds where no psg rate was computed: {psgNonCoveredSec} ({nonCovStr})')

    sns.histplot(psgAllBadIntervalLens*timeStep,discrete=False)
    plt.title('Distribution of interval lengths not covered by PSG')
    plt.show()

    nonCovH = radarNonCoveredSec // 3600
    minRem = radarNonCoveredSec % 3600
    nonCovMin = minRem // 60
    nonCovS = minRem % 60
    nonCovStr = f"{nonCovH:.0f} hours, {nonCovMin:.0f} minutes, {nonCovS:.1f} seconds"
    print(f'Seconds where no radar rate was computed: {radarNonCoveredSec} ({nonCovStr})')
    print('')

    sns.histplot(radarAllBadIntervalLens*timeStep,discrete=False)
    plt.title('Distribution of interval lengths not covered by radar')
    plt.show()

    print(f'              | With PSG | Without PSG ')
    print(f'---------------------------------------')
    print(f'With Radar    | {winsWithBoth:8.0f} | {winsWithNoPsgRadar:11.0f} ')
    print(f'---------------------------------------')
    print(f'Without Radar | {winsWithPsgNoRadar:8.0f} | {winsWithNeither:11.0f} ')
    print(f'---------------------------------------\n')

    print(f"Difference between PSG & Radar within +- {crtiDist}: {winsWithinCritDist} windows")
    print(f"    as proportion of time windows with both: {100*winsWithinCritDist/winsWithBoth: .1f}%")
    print(f"    as proportion of time windows with PSG:  {100*winsWithinCritDist/winsWithPSG: .1f}%")
    print(f"Difference between PSG & Radar within boundaries of uncertainty (+-{boundOfUncertain:.1f}): {winsRealClose} windows")
    print(f"    as proportion of time windows with both: {100*winsRealClose/winsWithBoth: .1f}%")
    print(f"    as proportion of time windows with PSG:  {100*winsRealClose/winsWithPSG: .1f}%")
    print('')
    # MSE
    mse = np.nanmean((fullPsgData - fullRadarData)**2)
    print(f"MSE between PSG and Radar:  {mse: .2f}")
    print("")

    mae = np.nanmean(np.abs(fullPsgData - fullRadarData))
    print(f"MAE between PSG and Radar:  {mae: .2f}")
    print("")

    mape = (1/winsWithBoth) * np.nansum(np.abs(fullPsgData - fullRadarData) / np.abs(fullPsgData)) * 100
    print(f"MAPE between PSG and Radar: {mape: .2f}%")
    print("")

    return
    ############################################################################################################################################
    ############################################################################################################################################

# The function is called here and will produce all of the results presented in the article:
for breathOrHeart in ["BR","HR"]:
    for radarChirps2use in ["all", 0]:
        compArticleResults(breathOrHeart=breathOrHeart, radarChirps2use=radarChirps2use)