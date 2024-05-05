import os
import sys
sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
from util.augmentationFunctions import get_augmentFunction_byName
from util.augmentationFunctions import get_nonZero_mean, get_samplingPDF
from util.dataPreprocessFunctions import get_sessionNameList
from util.trainingFunctions import get_dataset, train_NeuRA, train_tranditional_DA, test
import multiprocessing as mp
import pandas as pd
import numpy as np
import itertools

#
dataFolder = 'D:/Downloads/makin/indy'
saveFileName = f'./decodePerformance_all.csv'
interval = 0.064
timeLag = 20
NUM_PROCESS = 1
GPU_INDEX = 0

train_parameter = {
    'repeat_index': [i+1 for i in range(5)],
    'train_method': 'NeuRA', # ['traditional', 'NeuRA']
    'decoder_weight_init': ['init'], # ['init', 'inherit']
    'encoder_weight_trainable': ['frozen'], # ['frozen', 'trainable']
    'batchsize_pretrain': [256],
    'batchsize_finetune': [256],
    'epochs_pretrain': [100],
    'epochs_finetune': [100],
    'traindata_end_index': [5000, 4000, 3000],
    'testdata_start_index': [5000],
    'feature_name': ['channelBased'] # ['channelBased', 'unitBased]
}
augment_parameter = {
    'augment_method': 'mix', # 1 method in 'drop' / 'permute' / 'scale' / 'shift' / 'mix'
    'augment_frequency': ['sample'], # ['sample', 'batch']
    'drop_n': [2], # [n] for drop/mix
    'permute_n': [5], # [n] for permute/mix
    'scd_range': ['trainvstest'], # ['trainvstest', 'trainvstrain']
    'scale_channelForDistribution': ['all_channel'], # ['all_channel', 'single_channel'] for scale/mix
    'scale_sessionForDistributionCenter': ['all_session'], # ['all_session, 'single_session'] for scale/mix
    'scale_reverseDistributionCenter': ['reverse'], # ['reverse', 'remain'] for scale/mix
    'shift_channelForDistribution': ['all_channel'], # ['all_channel', 'single_channel'] for scale/mix
    'shift_sessionForDistributionCenter': ['all_session'], # ['all_session, 'single_session'] for scale/mix
    'shift_reverseDistributionCenter': ['reverse'], # ['reverse', 'remain'] for scale/mix
    'permute_probability': [0.5], # for mix
    'drop_probability': [0.5], # for mix
    'scale_probability': [0.5], # for mix
    'shift_probability': [0.5], # for mix
}


####################################################################################
# DONT MODIFY
####################################################################################
sessionNameList = get_sessionNameList(dataFolder)
# compress parameter dict
augment_parameter_include_keys = {
    'drop': ['augment_method', 'augment_frequency', 'drop_n'],
    'permute': ['augment_method', 'augment_frequency', 'permute_n'],
    'scale': ['augment_method', 'augment_frequency', 'scd_range', \
        'scale_channelForDistribution', 'scale_sessionForDistributionCenter', 'scale_reverseDistributionCenter'],
    'shift': ['augment_method', 'augment_frequency', 'scd_range', \
        'shift_channelForDistribution', 'shift_sessionForDistributionCenter', 'shift_reverseDistributionCenter'],
    'mix': ['augment_method', 'augment_frequency', 'drop_n', 'permute_n', \
        'scd_range', 'scale_channelForDistribution', 'scale_sessionForDistributionCenter', 'scale_reverseDistributionCenter', \
        'shift_channelForDistribution', 'shift_sessionForDistributionCenter', 'shift_reverseDistributionCenter', \
        'permute_probability', 'drop_probability', 'scale_probability', 'shift_probability'],
    'none': ['augment_method']
}
train_parameter_include_keys = {
    'traditional': ['repeat_index', 'train_method', \
        'batchsize_pretrain', 'batchsize_finetune', 'epochs_pretrain', 'epochs_finetune', \
        'traindata_end_index', 'testdata_start_index', 'feature_name'],
    'NeuRA': ['repeat_index', 'train_method', \
        'batchsize_pretrain', 'batchsize_finetune', 'epochs_pretrain', 'epochs_finetune', \
        'traindata_end_index', 'testdata_start_index', \
        'decoder_weight_init', 'encoder_weight_trainable', 'feature_name'],
}
augment_parameter = {
    k:v 
    for k,v in augment_parameter.items() 
    if k in augment_parameter_include_keys[augment_parameter['augment_method']]
    }
augment_parameter['augment_method'] = [augment_parameter['augment_method']]
train_parameter = {
    k:v 
    for k,v in train_parameter.items() 
    if k in train_parameter_include_keys[train_parameter['train_method']]
    }
train_parameter['train_method'] = [train_parameter['train_method']]

dataframeColumnName = ['sessionCount', 'sessionName', \
    'sessionCount_evaluation', 'sessionName_evaluation','r2_weighted', 'cc_weighted',\
    *train_parameter.keys(), *augment_parameter.keys()]

# work_function
def work_function(sessionCount:int, sessionName:str, train_param:dict, augment_param:dict):
    n_run = train_param['repeat_index']
    train_method = train_param['train_method']
    batchsize_pretrain = train_param['batchsize_pretrain']
    batchsize_finetune = train_param['batchsize_finetune']
    epochs_pretrain = train_param['epochs_pretrain']
    epochs_finetune = train_param['epochs_finetune']
    traindata_end_index = train_param['traindata_end_index']
    testdata_start_index = train_param['testdata_start_index']
    feature_name = train_param['feature_name']
    ####################################################################################
    if train_method == 'NeuRA':
        decoder_weight_init = train_param.get('decoder_weight_init')
        encoder_weight_trainable = train_param.get('encoder_weight_trainable')
    ####################################################################################
    augment_method = augment_param['augment_method']    
    augment_frequency = augment_param.get('augment_frequency')
    # drop
    n_drop = augment_param.get('drop_n')
    # permute
    n_permute = augment_param.get('permute_n')
    # scale
    scale_channelForDistribution = augment_param.get('scale_channelForDistribution')
    scale_sessionForDistributionCenter = augment_param.get('scale_sessionForDistributionCenter')
    scale_reverseDistributionCenter = augment_param.get('scale_reverseDistributionCenter')
    # shift
    shift_channelForDistribution = augment_param.get('shift_channelForDistribution')
    shift_sessionForDistributionCenter = augment_param.get('shift_sessionForDistributionCenter')
    shift_reverseDistributionCenter = augment_param.get('shift_reverseDistributionCenter')
    # mix    
    permute_probability = augment_param.get('permute_probability')
    drop_probability = augment_param.get('drop_probability')
    scale_probability = augment_param.get('scale_probability')
    shift_probability = augment_param.get('shift_probability')

    scd_range = augment_param.get('scd_range')

    ####################################################################################
    gpus = tf.config.list_physical_devices('GPU')
    gpu = gpus[GPU_INDEX]
    tf.config.set_visible_devices(gpu, 'GPU')
    tf.config.experimental.set_memory_growth(gpu, True)
    ####################################################################################
    # prepare datas
    x_train = np.zeros([0, 20, 96])
    y_train = np.zeros([0, 2])
    for i in range(sessionCount):
        _x_train, _, _y_train, _ = get_dataset(
            sessionName=sessionNameList[i],
            interval=interval,
            dataFolder=dataFolder,
            timeLag=timeLag,
            traindata_end_index=traindata_end_index,
            testdata_start_index=testdata_start_index,
            featureName=feature_name
        )
        x_train = np.concatenate([x_train, _x_train], axis=0)
        y_train = np.concatenate([y_train, _y_train], axis=0)

    

    ####################################################################################
    # augment function
    augment_function = get_augmentFunction_byName(name=augment_method, augment_freq=augment_frequency)
    
    if augment_method == 'drop':
        augmentFunc_simplify = lambda x: augment_function(x, n_drop)
    elif augment_method == 'permute':
        augmentFunc_simplify = lambda x: augment_function(x, n_permute)
    elif augment_method == 'scale':
        scaleMean, scaleStd = get_samplingPDF(
        method='scale', 
        sessionCount=sessionCount,
        channelForDistribution=scale_channelForDistribution, 
        reverseDistributionCenter=scale_reverseDistributionCenter, 
        sessionForDistributionCenter=scale_sessionForDistributionCenter, 
        scdRange=scd_range)
        sampleMean_nonZero = get_nonZero_mean(x_train[:, -1, :])
        augmentFunc_simplify = lambda x: augment_function(x, sampleMean_nonZero, scaleMean, scaleStd)
    elif augment_method == 'shift':
        shiftMean, shiftStd = get_samplingPDF(
        method='shift', 
        sessionCount=sessionCount,
        channelForDistribution=shift_channelForDistribution, 
        reverseDistributionCenter=shift_reverseDistributionCenter, 
        sessionForDistributionCenter=shift_sessionForDistributionCenter, 
        scdRange=scd_range)
        augmentFunc_simplify = lambda x: augment_function(x, shiftMean, shiftStd)
    elif augment_method == 'mix':
        scaleMean, scaleStd = get_samplingPDF(
            method='scale', 
            sessionCount=sessionCount,
            channelForDistribution=scale_channelForDistribution, 
            reverseDistributionCenter=scale_reverseDistributionCenter, 
            sessionForDistributionCenter=scale_sessionForDistributionCenter, 
            scdRange=scd_range)
        sampleMean_nonZero = get_nonZero_mean(x_train[:, -1, :])
        shiftMean, shiftStd = get_samplingPDF(
            method='shift', 
            sessionCount=sessionCount,
            channelForDistribution=shift_channelForDistribution, 
            reverseDistributionCenter=shift_reverseDistributionCenter, 
            sessionForDistributionCenter=shift_sessionForDistributionCenter,
            scdRange=scd_range)
        augmentFunc_simplify = lambda x: augment_function(x, sampleMean_nonZero, \
            scaleMean, scaleStd, shiftMean, shiftStd, n_drop, n_permute, \
            p_permute=permute_probability, p_drop=drop_probability, \
            p_scale=scale_probability, p_shift=shift_probability)
    else:
        raise ValueError('augment method name not in list')
            
    ####################################################################################
    # pre-train encoder
    tf.keras.backend.clear_session()    
    if train_method == 'traditional':
        kinematic_decoder = train_tranditional_DA(
            epochs_pretrain=epochs_pretrain,
            epochs_finetune=epochs_finetune,
            batchsize_pretrain=batchsize_pretrain,
            batchsize_finetune=batchsize_finetune,
            x_train=x_train,
            y_train=y_train,
            augmentFunction=augmentFunc_simplify
            )
    elif train_method == 'NeuRA':
        kinematic_decoder = train_NeuRA(
            epochs_pretrain=epochs_pretrain,
            epochs_finetune=epochs_finetune,
            batchsize_pretrain=batchsize_pretrain,
            batchsize_finetune=batchsize_finetune,
            x_train=x_train,
            y_train=y_train,
            augmentFunction=augmentFunc_simplify,
            decoderWeightInit=decoder_weight_init,
            encoderWeightFrozen=encoder_weight_trainable
        )
    
    ###################################################################
    # eval
    results = []
    for n_eval in range(sessionCount-1, 37):
        sessionCount_eval = n_eval + 1
        sessionName_eval = sessionNameList[n_eval]

        _, x_test, _, y_test = get_dataset(
            sessionName=sessionName_eval,
            interval=interval,
            dataFolder=dataFolder,
            timeLag=timeLag,
            traindata_end_index=traindata_end_index,
            testdata_start_index=testdata_start_index,
            featureName=feature_name
        )

        # calculate r-square score
        r2 = test(kinematic_decoder, x_test, y_test, performanceIndex='r2')
        cc = test(kinematic_decoder, x_test, y_test, performanceIndex='cc')
        
        # print and save results
        print(f'{n_run} {sessionCount=} {sessionCount_eval=} {r2=} {cc=} {augment_param=} {train_param=}')
        results.append(
            [sessionCount, sessionName, sessionCount_eval, sessionName_eval, r2, cc, \
                *train_param.values(), *augment_param.values()]
            )
    return results


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # create results file 
    if not os.path.isfile(saveFileName):
        df = pd.DataFrame(columns=dataframeColumnName)
        df.to_csv(saveFileName, index=False)
    
    # run sub process
    for n, sessionName in enumerate(sessionNameList):
        sessionCount = n + 1

        train_parameter_keys = train_parameter.keys()
        train_parameter_values = train_parameter.values()
        augment_parameter_keys = augment_parameter.keys()
        augment_parameter_values = augment_parameter.values()

        multiple_results = []
        with mp.Pool(processes=NUM_PROCESS) as pool:
            for train_param in itertools.product(*train_parameter_values):
                for augment_param in itertools.product(*augment_parameter_values):
                    train_param_dict = dict(zip(train_parameter_keys, train_param))
                    augment_param_dict = dict(zip(augment_parameter_keys, augment_param))
                    multiple_results.append(
                        pool.apply_async(
                            work_function, 
                            (sessionCount, sessionName, train_param_dict, augment_param_dict)
                            )
                        )
            df = [res for results in multiple_results for res in results.get()]

        df = pd.DataFrame(df, columns=dataframeColumnName)
        df.to_csv(
            path_or_buf=saveFileName, 
            index=False, 
            header=False, 
            mode='a'
            )
