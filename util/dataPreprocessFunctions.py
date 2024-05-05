import numpy as np
import pandas as pd
import os
from scipy.interpolate import CubicSpline
import scipy.signal as signal

def add_lags(arr: np.ndarray, timeLag:int = 20) -> np.ndarray:
    assert len(arr.shape) == 2
    arr_withLags = np.pad(arr, [[timeLag-1, 0], [0, 0]])
    arr_withLags = np.vstack([np.roll(arr_withLags[np.newaxis, :, :], i, axis=1) for i in range(timeLag-1, -1, -1)])
    arr_withLags = arr_withLags.transpose(1, 0, 2)
    arr_withLags = arr_withLags[timeLag-1:, :, :]
    return arr_withLags

def get_sessionNameList(dataFolder):
    infomationFilePath = os.path.join(dataFolder, 'Informations.csv')
    df_informations = pd.read_csv(infomationFilePath)

    return df_informations['sessionName'].to_numpy()

def get_spiketrain(sessionName:str, interval:float, dataFolder:str, featureName='channelBased') -> np.ndarray:
    assert featureName in ['channelBased', 'unitBased'], 'sugnal type not in access list'

    kinematicFolder = os.path.join(dataFolder, 'kinematics')
    spikeTrainFolder = os.path.join(dataFolder, 'spikeTrains')
    infomationFilePath = os.path.join(dataFolder, 'Informations.csv')

    df_kinematic = pd.read_parquet(os.path.join(kinematicFolder, sessionName + '.parq.gz'))
    df_spikeTrain = pd.read_parquet(os.path.join(spikeTrainFolder, sessionName + '.parq.gz'))
    df_informations = pd.read_csv(infomationFilePath)

    # get Time
    timestamp = sorted(df_kinematic['timestamp'])
    targetTimestamp = np.arange(timestamp[0], timestamp[-1], interval)
    targetTimestamp = np.round(targetTimestamp, decimals=4)

    # Session informations
    max_channelCount = df_informations[df_informations['sessionName'] == sessionName]['channelLength_m1'].to_numpy()[0]
    max_unitCount = df_informations[df_informations['sessionName'] == sessionName]['unitLength'].to_numpy()[0]

    match featureName:
        case 'channelBased':
            spikeTrains = df_spikeTrain[df_spikeTrain['cortex'] == 'm1']\
                .groupby(['channel']).agg(list).sort_values('channel')['spikeTime']
            # calculate data    
            spikeCount_featureCount = max_channelCount
        case 'unitBased':
            df_spikeTrain['unitIdx'] = df_spikeTrain['channel'] * max_unitCount + df_spikeTrain['unit']
            spikeTrains = df_spikeTrain[df_spikeTrain['cortex'] == 'm1']\
                .groupby(['unitIdx']).agg(list).sort_values('unitIdx')['spikeTime']
            
            spikeCount_featureCount = max_channelCount * max_unitCount
        case _:
            raise ValueError(f'\'signalType\' value error')
            

    spikeCount = np.zeros([len(targetTimestamp)-1, spikeCount_featureCount])
          
    for idx in spikeTrains.index:        
        hist, bin_edges = np.histogram(spikeTrains[idx], bins=targetTimestamp)
        spikeCount[:, idx] = hist

    return spikeCount

def get_timestamp(sessionName:str, interval:float, dataFolder:str) -> np.ndarray:
    kinematicFolder = os.path.join(dataFolder, 'kinematics')

    df_kinematic = pd.read_parquet(os.path.join(kinematicFolder, sessionName + '.parq.gz'))

    timestamp = sorted(df_kinematic['timestamp'])
    targetTimestamp = np.arange(timestamp[0], timestamp[-1], interval)
    targetTimestamp = np.round(targetTimestamp, decimals=4)
    
    return targetTimestamp[1:]

def apply_gaussian(spiketrain:np.ndarray, interval:float) -> np.ndarray:
    kern_sd_ms = 40
    kern_sd = int(round(kern_sd_ms / (interval*1000)))
    window = signal.gaussian(kern_sd * 6 + 1, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')

    spikeCount_gaussian = np.apply_along_axis(filt, 0, spiketrain)
    return spikeCount_gaussian

def get_kinematic(sessionName:str, interval:float, dataFolder:str) -> np.ndarray:
    kinematicFolder = os.path.join(dataFolder, 'kinematics')

    df_kinematic = pd.read_parquet(os.path.join(kinematicFolder, sessionName + '.parq.gz'))
    timestamp = sorted(df_kinematic['timestamp'])
    targetTimestamp = np.arange(timestamp[0], timestamp[-1], interval)
    targetTimestamp = np.round(targetTimestamp, decimals=4)
    
    axisList = ['x', 'y']
    CS = [CubicSpline(timestamp, df_kinematic[('finger_pos', axis)]) for axis in axisList]
    kinematic_position = np.concatenate([cs(targetTimestamp, 0)[:, None] for cs in CS], axis=1)
    kinematic_velocity = np.concatenate([cs(targetTimestamp, 1)[:, None] for cs in CS], axis=1)
    kinematic_accelerate = np.concatenate([cs(targetTimestamp, 2)[:, None] for cs in CS], axis=1)
    kinematic = np.concatenate([kinematic_position, kinematic_velocity, kinematic_accelerate], axis=1)
    kinematic = kinematic[:-1, :]
    return kinematic

def get_sessionCount(sessionName, dataFolder):
    kinematicFolder = os.path.join(dataFolder, 'kinematics')

    fileList = sorted([a.split('.')[0] for a in os.listdir(kinematicFolder)])
    for n, f in enumerate(fileList):
        if f == sessionName:
            return n+1
    return -1


