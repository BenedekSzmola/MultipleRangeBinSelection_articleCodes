# Import External Packages
import copy
## load
import pickle  # to save the large measurement data
## calculation
import numpy as np

# Import internal packages
from helperFunctions import normDataTo_0_1
#####################################################

def readSaveFile(file_name):
    # Load Data
    with open(file_name, 'rb') as load_file:
        try:
            while True:
                file_info = (pickle.load(load_file))  # Information about measurement
                radar_var = (pickle.load(load_file))  # Configuration of radar (the same as constants.py)
                synchro_info = (pickle.load(load_file)) # Synchronsation Informationen
                measurement_data = (pickle.load(load_file)) # Measurement data of the PSG and all 3 Radars
                print('----------------File read----------------')
        except:
            print("EoF")

    return file_info,radar_var,synchro_info,measurement_data

def readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,srate,epochLen,epochInput,useHann=True,chirp=np.array([0]),chirpSumMethod="mean",doUnwrap=True,norm01=False,removeDC=False):
    def deinterleaveRadarData(rawInputData,TX,RX):
        ## Deinterleaving the Radar Data
        num_meas = rawInputData.shape[0]  # number of measurements in data. each subarray is one measurement in an interleaved int 16 format
        data_raw = rawInputData.flatten()
        data_raw = np.reshape(data_raw, (int(len(data_raw)/4),2,2)) # creates a 2x2 subarray 
        data_raw = data_raw.transpose(0,2,1) #switches the places for data in the 2x2 array to revoke the interleaved structure
        data_raw = np.reshape(data_raw,(len(data_raw)*2,2)) # reshapes everything into size 2 subarrays which represent real and imaginary part of the complex number
        data_raw = data_raw.transpose() # transposes everything to get two large subarray. the first is every imaginary part and the seconds every real part
        data_raw = 1j*data_raw[0] + data_raw[1] # combining real and imaginary parts for everything at once => array with complex numbers along the time axis
        data_comp_all = np.reshape(data_raw, (num_meas,radarSettings['radar_loop_num'],radarSettings['radar_tx_num'],radarSettings['radar_rx_num'],radarSettings['radar_adcsamp_num'])) # reshaping complex numbers back to individual measurmements

        ## Converting chirps to range representations
        data_comp = data_comp_all[:,:,TX,RX]

        return data_comp
    
    ## Radar        
    TX = 0  # TX =  Transmitting Antenna -> We only dont use angle information here so we only use the same antenna configuration
    RX = 0  # RX =  Receiving Antenna

    if np.isscalar(chirp):
        chirp = np.array([chirp])
    
    epochLen_inSamples = int(np.floor(epochLen * srate))
    
    ## Extracting Radar data
    timestamps = measurement_data[radar_idx][0]
    rawInputData = measurement_data[radar_idx][1]

    fullSamplesLen = len(timestamps)
    measLenInSec = timestamps[-1] - timestamps[0]
    
    if 'timeStart' in epochInput:
        timeStart     = epochInput['timeStart']
        timeEnd       = epochInput['timeEnd']
        epochStepSize = epochInput['epochStepSize']

        epochStepSize_inSamples = int(np.floor(epochStepSize * srate))

        # Selecting the interval to be extracted (timeStart - timeEnd)
        if (timeStart > timeEnd) or ((timeEnd - timeStart) < epochLen):
            raise Exception("Inputs for timeStart and timeEnd are incorrect! (either timeStart is after timeEnd, or the difference between them is less then epochLen)")

        if np.isinf(timeStart) or (timeStart > (measLenInSec - epochLen)):
            raise Exception("The input for timeStart is incorrect! (either infinity or too close to end of recording)")

        timeStart_inSamples = np.argmin(np.abs(timestamps - (timeStart + timestamps[0])))
        if timeStart_inSamples < 0:
            print('Time start cannot be before the first sample! Setting to first sample')
            timeStart_inSamples = 0

        if np.isinf(timeEnd):
            timeEnd_inSamples = copy.deepcopy(fullSamplesLen)
        else:
            timeEnd_inSamples = np.argmin(np.abs(timestamps - (timeEnd + timestamps[0]))) + 1
            if timeEnd_inSamples > fullSamplesLen:
                print('Time end cannot be after the last sample! Setting to the last sample')
                timeEnd_inSamples = copy.deepcopy(fullSamplesLen)

        epochStartInds = np.arange(timeStart_inSamples, timeEnd_inSamples-epochLen_inSamples+1, epochStepSize_inSamples)

    elif 'epochStarts' in epochInput:
        epochStarts = epochInput['epochStarts']
        if np.isscalar(epochStarts):
            epochStarts = np.array([epochStarts])
        
        if any(epochStarts > (timestamps[-1] - epochLen)):
            raise Exception("At least one of the epochStarts is incorrect! (is too close to the end of the recording)")

        epochStartInds = np.zeros(len(epochStarts),dtype=int)

        for epochi in range(len(epochStarts)):
            epochStartInds[epochi] = np.argmin(np.abs(timestamps - epochStarts[epochi]))
    
    else: 
        raise Exception("False input for parameter 'epochInputs'!")
  

    numEpochs = len(epochStartInds)

    timestampEpochs = [0]*numEpochs
    exactEpochStartTimestamps = np.zeros(numEpochs)
    magnitudeEpochs = [0]*numEpochs
    phaseEpochs     = [0]*numEpochs

    for epochi,currEpochStart in enumerate(epochStartInds):
        currEpochEnd = currEpochStart + epochLen_inSamples

        timestampEpochs[epochi] = timestamps[currEpochStart:currEpochEnd]
        exactEpochStartTimestamps[epochi] = timestampEpochs[epochi][0]

        dataCompEpoch = deinterleaveRadarData(rawInputData[currEpochStart:currEpochEnd],TX,RX)

        if useHann:
            dataCompEpoch = dataCompEpoch * np.hanning(radarSettings['radar_adcsamp_num'])
        
        rangeDataEpoch = np.fft.fft(dataCompEpoch)

        tempMagnitude = np.zeros((len(chirp),radarSettings['radar_adcsamp_num'],len(timestampEpochs[epochi])))
        tempPhase     = np.zeros((len(chirp),radarSettings['radar_adcsamp_num'],len(timestampEpochs[epochi])))
        for chirpi,chirpNum in enumerate(chirp):
            tempMagnitude[chirpi,:,:] = np.abs(rangeDataEpoch[:,chirpNum,:].transpose())

            tempPhase[chirpi,:,:] = np.angle(rangeDataEpoch[:,chirpNum,:].transpose())
            if doUnwrap: 
                tempPhase[chirpi,:,:] = np.unwrap(tempPhase[chirpi,:,:])

            tempPhase[chirpi,:,:] = (tempPhase[chirpi,:,:]* radarSettings['speed_of_light']) / (radarSettings['radar_freq_min'] * 4*radarSettings['pi'] )  * 1000  # Calculating actual distance and converting from meter into mm

            if removeDC:
                tempPhase[chirpi,:,:] = tempPhase[chirpi,:,:] - np.mean(tempPhase[chirpi,:,:], axis=1, keepdims=True)

        if chirpSumMethod == "mean":
            magnitudeEpochs[epochi] = np.mean(tempMagnitude, axis=0)
            phaseEpochs[epochi]     = np.mean(tempPhase, axis=0)
            
        elif chirpSumMethod == "median":
            magnitudeEpochs[epochi] = np.median(tempMagnitude, axis=0)
            phaseEpochs[epochi]     = np.median(tempPhase, axis=0)

        if norm01:
            magnitudeEpochs[epochi] = normDataTo_0_1(magnitudeEpochs[epochi])
            
            phaseEpochs[epochi] = normDataTo_0_1(phaseEpochs[epochi])

    return timestampEpochs,phaseEpochs,magnitudeEpochs,exactEpochStartTimestamps,np.round(exactEpochStartTimestamps-timestamps[0]).astype("int")


def readFullRefData(measurement_data,sensor_idx):

    ## Extracting reference sensor data
    timestamps = measurement_data[sensor_idx][0]    
    data = measurement_data[sensor_idx][1]    

    print('-----Successfully read full reference data-----')

    return timestamps, data


def makeRefDataEpochs(timestamps,data,srate,epochLen,epochInput,norm01=False):
    epochLen_inSamples = int(np.floor(epochLen * srate))
    
    if 'radarEpochStarts' in epochInput:
        radarEpochStarts = epochInput['radarEpochStarts']
        epochStartInds = np.zeros(len(radarEpochStarts),dtype=int)

        for i in range(len(radarEpochStarts)):
            epochStartInds[i] = np.argmin(np.abs(timestamps - radarEpochStarts[i]))

    elif 'timeStart' in epochInput:
        timeStart     = epochInput['timeStart']
        timeEnd       = epochInput['timeEnd']
        epochStepSize = epochInput['epochStepSize']

        epochStepSize_inSamples = int(np.floor(epochStepSize * srate))

        # Selecting the interval to be extracted (timeStart - timeEnd)
        timeStart_inSamples = np.argmin(np.abs(timestamps - (timeStart + timestamps[0])))
        if timeStart_inSamples < 0:
            print('Time start cannot be before the first sample! Setting to first sample')
            timeStart_inSamples = 0

        if np.isinf(timeEnd):
            timeEnd_inSamples = data.shape[0]
        else:
            timeEnd_inSamples = np.argmin(np.abs(timestamps - (timeEnd + timestamps[0]))) + 1
            if timeEnd_inSamples > data.shape[0]:
                print('Time end cannot be after the last sample! Setting to the last sample')
                timeEnd_inSamples = data.shape[0]

        epochStartInds = np.arange(timeStart_inSamples, timeEnd_inSamples-epochLen_inSamples+1, epochStepSize_inSamples)

    elif 'epochStarts' in epochInput:
        epochStarts = epochInput['epochStarts']
        if np.isscalar(epochStarts):
            epochStarts = np.array([epochStarts])
        
        epochStartInds = np.zeros(len(epochStarts),dtype=int)

        for epochi in range(len(epochStarts)):
            epochStartInds[epochi] = np.argmin(np.abs(timestamps - epochStarts[epochi]))
    
    else: 
        raise Exception("False input for parameter 'epochInputs'!")

    numEpochs = len(epochStartInds)

    timestampEpochs = [0]*numEpochs
    exactEpochStartTimestamps = np.zeros(numEpochs)
    dataEpochs   = [0]*numEpochs
    
    for epochi,currEpochStart in enumerate(epochStartInds):
        currEpochEnd = currEpochStart + epochLen_inSamples

        timestampEpochs[epochi] = timestamps[currEpochStart:currEpochEnd]
        exactEpochStartTimestamps[epochi] = timestampEpochs[epochi][0]

        dataEpochs[epochi] = data[currEpochStart:currEpochEnd]

        if norm01:            
            dataEpochs[epochi] = normDataTo_0_1(dataEpochs[epochi])

    return timestampEpochs,dataEpochs,exactEpochStartTimestamps,np.round(exactEpochStartTimestamps-timestamps[0]).astype("int")
