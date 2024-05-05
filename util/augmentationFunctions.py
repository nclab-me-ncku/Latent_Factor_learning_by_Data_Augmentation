import tensorflow as tf
import numpy as np
import pandas as pd

#%%
def Electrode_Permuting_perSample(x, n_permute:int):
    # x.shape [N, 20, 96]
    bs, _, num_electrode = x.shape
    positions = np.array([np.random.permutation(num_electrode) for _ in range(bs)], dtype=np.int32)
    pre_position = positions[:, :n_permute]
    new_position = np.apply_along_axis(np.random.permutation, axis=1, arr=pre_position)

    #
    bs_indices = np.broadcast_to(np.arange(bs), [n_permute, bs]).T.flatten()
    x_out = x.numpy()
    x_out[bs_indices, :, pre_position.flatten()] = x_out[bs_indices, :, new_position.flatten()]

    return x_out

def Electrode_Dropping_perSample(x, n_drop:int):
    # x.shape [N, 20, 96]
    bs, _, num_electrode = x.shape
    _mask = np.array([np.random.permutation(num_electrode) for _ in range(bs)])
    mask = tf.cast(_mask>=n_drop, dtype=tf.float32)

    x_out = tf.transpose(x, perm=[1, 0, 2]) * mask
    x_out = tf.transpose(x_out, perm=[1, 0, 2])

    return x_out

def SCD_Scaling_perSample(x, sampleMean_nonZero, distributionMean, distributionStd):
    # x.shape [N, 20, 96]
    bs, _, num_electrode = x.shape
    #
    distributionMean = tf.clip_by_value(distributionMean, clip_value_min=0, clip_value_max=tf.float32.max)
    #
    scale = tf.random.normal(shape=[bs, num_electrode], mean=0, stddev=1, dtype=tf.float32)
    scale = scale * distributionStd + distributionMean

    scale = tf.clip_by_value(scale, clip_value_min=0, clip_value_max=tf.float32.max)
    
    x_out = tf.transpose(x, perm=[1, 0, 2])
    x_out = tf.pow(x_out, scale) * tf.exp(sampleMean_nonZero * (1 - scale))
    x_out = tf.transpose(x_out, perm=[1, 0, 2])
    
    return x_out

def SCD_Shifting_perSample(x, distributionMean, distributionStd):
    # x.shape [N, 20, 96]
    bs, _, num_electrode = x.shape

    shift = tf.random.normal(shape=[bs, num_electrode], mean=0, stddev=1, dtype=tf.float32)
    shift = shift * distributionStd + distributionMean

    x_out = tf.transpose(x, perm=[1, 0, 2])
    x_out = x_out * tf.exp(shift)
    x_out = tf.transpose(x_out, perm=[1, 0, 2])

    return x_out

def mix_perSample(x, sampleMean_nonZero, scaleMean, scaleStd, shiftMean, shiftStd, n_drop, n_permute, \
    p_permute, p_drop, p_scale, p_shift):
    choice_permute = np.random.choice([False, True], 1, p=[1-p_permute, p_permute])
    choice_drop = np.random.choice([False, True], 1, p=[1-p_drop, p_drop])
    choice_scale = np.random.choice([False, True], 1, p=[1-p_scale, p_scale])
    choice_shift = np.random.choice([False, True], 1, p=[1-p_shift, p_shift])
    
    x = tf.cast(x, dtype=tf.float32)

    if choice_permute:
        x = Electrode_Permuting_perSample(x, n_permute=n_permute)
    if choice_drop:
        x = Electrode_Dropping_perSample(x, n_drop=n_drop)
    if choice_scale:
        x = SCD_Scaling_perSample(x, sampleMean_nonZero=sampleMean_nonZero, distributionMean=scaleMean, distributionStd=scaleStd)
    if choice_shift:
        x = SCD_Shifting_perSample(x, distributionMean=shiftMean, distributionStd=shiftStd)

    return x

#%%
def Electrode_Permuting_perBatch(x, n_permute:int):
    bs, _, num_electrode = x.shape

    choice_channel = tf.range(0, num_electrode)
    choice_channel = tf.random.shuffle(choice_channel)
    choice_channel = choice_channel[:n_permute]

    new_choice_channel = tf.random.shuffle(choice_channel)

    # new channel order
    channel_index = np.arange(num_electrode)
    channel_index[choice_channel.numpy()] = new_choice_channel.numpy()


    channels = tf.unstack(x, axis=-1)
    x_out = tf.stack([channels[k] for k in channel_index], axis=-1)

    return x_out

def Electrode_Dropping_perBatch(x, n_drop:int):
    bs, _, num_electrode = x.shape

    choice_channel = tf.range(0, num_electrode)
    choice_channel = tf.random.shuffle(choice_channel)
    choice_channel = choice_channel[:n_drop].numpy()

    mask = tf.ones([num_electrode]).numpy()
    mask[choice_channel] = 0
    
    x_out = x * tf.cast(mask, dtype=tf.float32)

    return x_out

def SCD_Scaling_perBatch(x, sampleMean_nonZero, distributionMean, distributionStd):
    bs, _, num_electrode = x.shape

    distributionMean = tf.clip_by_value(distributionMean, clip_value_min=0, clip_value_max=tf.float32.max)

    scale = tf.random.normal(shape=[num_electrode], mean=0, stddev=1, dtype=tf.float32)
    scale = scale * distributionStd + distributionMean

    scale = tf.clip_by_value(scale, clip_value_min=0, clip_value_max=tf.float32.max)
    
    x_out = tf.pow(x, scale) * tf.exp(sampleMean_nonZero * (1 - scale))
    
    return x_out

def SCD_Shifting_perBatch(x, distributionMean, distributionStd):
    bs, _, num_electrode = x.shape
    
    shift = tf.random.normal(shape=[num_electrode], mean=0, stddev=1, dtype=tf.float32)
    shift = shift * distributionStd + distributionMean

    x_out = x * tf.exp(shift)

    return x_out

def mix_perBatch(x, sampleMean_nonZero, scaleMean, scaleStd, shiftMean, shiftStd, n_drop, n_permute, \
    p_permute, p_drop, p_scale, p_shift):
    choice_permute = np.random.choice([False, True], 1, p=[1-p_permute, p_permute])
    choice_drop = np.random.choice([False, True], 1, p=[1-p_drop, p_drop])
    choice_scale = np.random.choice([False, True], 1, p=[1-p_scale, p_scale])
    choice_shift = np.random.choice([False, True], 1, p=[1-p_shift, p_shift])
    
    x = tf.cast(x, dtype=tf.float32)

    if choice_permute:
        x = Electrode_Permuting_perBatch(x, n_permute=n_permute)
    if choice_drop:
        x = Electrode_Dropping_perBatch(x, n_drop=n_drop)
    if choice_scale:
        x = SCD_Scaling_perBatch(x, sampleMean_nonZero=sampleMean_nonZero, distributionMean=scaleMean, distributionStd=scaleStd)
    if choice_shift:
        x = SCD_Shifting_perBatch(x, distributionMean=shiftMean, distributionStd=shiftStd)

    return x

#%%
def get_samplingPDF(method:str, sessionCount:int, \
    channelForDistribution:str, reverseDistributionCenter:str, sessionForDistributionCenter:str, \
    scdRange:str):
    assert method in ['scale', 'shift'], 'we do not support the method you input'
    assert sessionCount in [i+1 for i in range(37)], 'session range support from 1 to 37'
    assert scdRange in ['trainvstest', 'trainvstrain']
    assert channelForDistribution in ['all_channel', 'single_channel']
    assert reverseDistributionCenter in ['reverse', 'remain']
    assert sessionForDistributionCenter in ['all_session', 'single_session']

    scdRangeFile = f'./results/dataAugmentation/distribtionAugmentationParameters_{scdRange}.csv'

    df_range = pd.read_csv(scdRangeFile)
    df_range['meanSession'] = df_range.apply(lambda x: x['mean'] if np.isnan(x['meanSession']) else x['meanSession'], axis=1)
    g = df_range.groupby(['randomMethod', 'sessionCount', 'channel']).mean()

    match method:
        case 'scale':
            match channelForDistribution:
                case 'all_channel':
                    match sessionForDistributionCenter:
                        case 'all_session':
                            scaleMean = g.loc[('scale', sessionCount, 'all')]['mean']
                        case 'single_session':
                            scaleMean = g.loc[('scale', sessionCount, 'all')]['meanSession']
                    scaleStd = g.loc[('scale', sessionCount, 'all')]['sd']
                    # remove outliers
                    scaleStd = 0 if scaleStd > 1 else scaleStd
                case 'single_channel':
                    match sessionForDistributionCenter:
                        case 'all_session':
                            scaleMean = [(g.loc[('scale', sessionCount, str(ch+1))])['mean'] for ch in range(CHANNEL_COUNT)]
                        case 'single_session':
                            scaleMean = [(g.loc[('scale', sessionCount, str(ch+1))])['meanSession'] for ch in range(CHANNEL_COUNT)]
                    scaleMean = np.array(scaleMean)
                    scaleStd = np.array(
                        [(g.loc[('scale', sessionCount, str(ch+1))])['sd'] for ch in range(CHANNEL_COUNT)]
                        )           

                    # remove outliers
                    scaleStd[scaleStd > 1] = 0
            if reverseDistributionCenter == 'reverse':
                scaleMean = 1 + (1 - scaleMean)
            # 
            scaleMean = tf.convert_to_tensor(scaleMean, dtype=tf.float32)
            scaleStd = tf.convert_to_tensor(scaleStd, dtype=tf.float32)

            return scaleMean, scaleStd
        case 'shift':
            match channelForDistribution:
                case 'all_channel':
                    match sessionForDistributionCenter:
                        case 'all_session':
                            shiftMean = g.loc[('shift', sessionCount, 'all')]['mean']
                        case 'single_session':
                            shiftMean = g.loc[('shift', sessionCount, 'all')]['meanSession']
                    shiftStd = g.loc[('shift', sessionCount, 'all')]['sd']
                    # remove outliers
                    shiftStd = 0 if shiftStd > 1 else shiftStd
                case 'single_channel':
                    match sessionForDistributionCenter:
                        case 'all_session':
                            shiftMean = [(g.loc[('shift', sessionCount, str(ch+1))])['mean'] for ch in range(CHANNEL_COUNT)]
                        case 'single_session':
                            shiftMean = [(g.loc[('shift', sessionCount, str(ch+1))])['meanSession'] for ch in range(CHANNEL_COUNT)]
                    shiftMean = np.array(shiftMean)
                    shiftStd = np.array(
                        [(g.loc[('shift', sessionCount, str(ch+1))])['sd'] for ch in range(CHANNEL_COUNT)]
                        )
                    # remove outliers
                    shiftStd[shiftStd > 1] = 0
            if reverseDistributionCenter == 'reverse':
                shiftMean = 0 + (0 - shiftMean)
            shiftMean = tf.convert_to_tensor(shiftMean, dtype=tf.float32)
            shiftStd = tf.convert_to_tensor(shiftStd, dtype=tf.float32)

            return shiftMean, shiftStd

def get_nonZero_mean(spikecount):

    nonZero_mean = [
        np.mean(spikecount_perChannel[spikecount_perChannel != 0]) 
        if len(spikecount_perChannel[spikecount_perChannel != 0]) else 0 
        for spikecount_perChannel in spikecount[:, :].T
        ]
    nonZero_mean = np.array(nonZero_mean)
    nonZero_mean = np.nan_to_num(nonZero_mean)
    nonZero_mean = tf.convert_to_tensor(nonZero_mean, dtype=tf.float32)

    return nonZero_mean

#%%
def get_augmentFunction_byName(name:str, augment_freq:str):
    assert name in ['none', 'drop', 'permute', 'scale', 'shift', 'mix']

    match name:
        case 'none':
            return lambda x: x
        case 'drop':
            if augment_freq == 'batch':
                return Electrode_Dropping_perBatch
            elif augment_freq == 'sample':
                return Electrode_Dropping_perSample
        case 'permute':
            if augment_freq == 'batch':
                return Electrode_Permuting_perBatch
            elif augment_freq == 'sample':
                return Electrode_Permuting_perSample
        case 'scale':
            if augment_freq == 'batch':
                return SCD_Scaling_perBatch
            elif augment_freq == 'sample':
                return SCD_Scaling_perSample
        case 'shift':
            if augment_freq == 'batch':
                return SCD_Shifting_perBatch
            elif augment_freq == 'sample':
                return SCD_Shifting_perSample
        case 'mix':
            if augment_freq == 'batch':
                return mix_perBatch
            elif augment_freq == 'sample':
                return mix_perSample
    
