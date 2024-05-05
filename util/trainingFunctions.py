from util.dataPreprocessFunctions import add_lags, get_kinematic, get_spiketrain, apply_gaussian
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from util.neuralDecoderModels import LSTM_neuralDecoder, extractor

def get_dataset(
    sessionName:str, 
    interval:float, 
    dataFolder:str, 
    featureName:str, 
    timeLag:int, 
    traindata_end_index:int,
    testdata_start_index:int
    ):
    '''
    回傳訓練以及測試資料

    Parameters
    ----------
    splitIndex : int
        訓練資料以及測試資料之切割位置
    
    Returns
    -------
    (x_train, x_test, y_train, y_test)
    '''
    spikeCount = get_spiketrain(sessionName, interval, dataFolder, featureName=featureName)
    spikeCount_gaussian = apply_gaussian(spikeCount, interval)
    kinematic = get_kinematic(sessionName, interval, dataFolder)[:, 2:4]
    kinematic_mean = np.mean(kinematic[:traindata_end_index], axis=0)
    kinematic_std = np.std(kinematic[:traindata_end_index], axis=0)
    kinematic_norm = (kinematic - kinematic_mean) / kinematic_std
    spikeCount_gaussian_withLags = add_lags(spikeCount_gaussian, timeLag)
    x_train, _, x_test = np.split(spikeCount_gaussian_withLags, [traindata_end_index, testdata_start_index], axis=0)
    y_train, _, y_test = np.split(kinematic_norm, [traindata_end_index, testdata_start_index], axis=0)


    return x_train, x_test, y_train, y_test

def get_tfDataset(x, y, batchsize:int):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
    dataset = dataset.shuffle(buffer_size=5000).batch(batch_size=batchsize)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def train_tranditional_DA(epochs_pretrain, epochs_finetune, batchsize_pretrain, batchsize_finetune, x_train, y_train, augmentFunction):
    tfDataset = get_tfDataset(x_train, y_train, batchsize_pretrain)

    # create augmented data
    x_augmented = [augmentFunction(x) for x, y in tfDataset]
    x_augmented = np.concatenate(x_augmented, axis=0)
    print(x_augmented.shape)
    # combine datas
    _x_train = np.concatenate([x_train, x_augmented], axis=0)
    _y_train = np.concatenate([y_train, y_train], axis=0)
    print(_x_train.shape, _y_train.shape)
    # train model
    model = LSTM_neuralDecoder(2)
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model.fit(x=_x_train, y=_y_train, batch_size=batchsize_finetune, epochs=epochs_finetune, shuffle=True, verbose=0)
    exit()
    return model

def train_tranditional(epochs, batchsize, x_train, y_train):
    model = LSTM_neuralDecoder(2)
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model.fit(x=x_train, y=y_train, batch_size=batchsize, epochs=epochs, shuffle=True, verbose=0)
    return model

def finetune_tranditional(model, epochs, batchsize, x_train, y_train):
    model.fit(x=x_train, y=y_train, batch_size=batchsize, epochs=epochs, shuffle=True, verbose=0)
    return model

def train_NeuRA(epochs_pretrain, epochs_finetune, batchsize_pretrain, batchsize_finetune, x_train, y_train, augmentFunction,\
    decoderWeightInit:str='init', encoderWeightFrozen:str='frozen'):
    x_featureCount = x_train.shape[-1]

    tfDataset = get_tfDataset(x_train, y_train, batchsize_pretrain)
    # stage 1
    model = extractor(2)
    model.compile(optimizer='adam', run_eagerly=True)
    for _ in range(epochs_pretrain):
        for x, y in tfDataset:
            inputs = {
                'x1': augmentFunction(x),
                'x2': augmentFunction(x),
                'y': y
            }
            model.train_on_batch(x=inputs)
    # stage 2
    kinematic_decoder = LSTM_neuralDecoder(2)    
    kinematic_decoder.build([None, 20, x_featureCount])
    kinematic_decoder.encoder.set_weights(model.encoder.get_weights())
    if encoderWeightFrozen == 'frozen':
        kinematic_decoder.encoder.trainable = False
    else:
        kinematic_decoder.encoder.trainable = True
    
    if decoderWeightInit == 'init':
        pass
    else:
        kinematic_decoder.decoder.set_weights(model.neural_decoder.get_weights())
        kinematic_decoder.encoder.trainable = True
        
    kinematic_decoder.compile(optimizer='adam', loss='mse', run_eagerly=False)
    kinematic_decoder.fit(x=x_train, y=y_train, batch_size=batchsize_finetune, epochs=epochs_finetune, shuffle=True, verbose=0)
    return kinematic_decoder

def test(model, x_test, y_test, performanceIndex='r2'):
    if performanceIndex == 'r2':
        pred_y = [model(x_test_split) for x_test_split in np.array_split(x_test, 20)]
        pred_y = np.concatenate(pred_y, axis=0)
        r2 = r2_score(y_pred=pred_y, y_true=y_test, multioutput='variance_weighted')
        return r2
    elif performanceIndex == 'cc':
        pred_y = [model(x_test_split) for x_test_split in np.array_split(x_test, 20)]
        pred_y = np.concatenate(pred_y, axis=0)

        cc = [np.corrcoef(pred, test)[0, 1] for pred, test in zip(pred_y.T, y_test.T)]
        var = [np.var(test) for test in y_test.T]   
        cc_weighted = np.average(cc, weights=var)
        return cc_weighted

def test_trajectory(model, x_test):
    pred_y = [model(x_test_split) for x_test_split in np.array_split(x_test, 20)]
    pred_y = np.concatenate(pred_y, axis=0)
    return pred_y