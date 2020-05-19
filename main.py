import pandas_datareader as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from statsmodels.robust import mad
from scipy import signal
import data_reader, features
from alpha_vantage.timeseries import TimeSeries 
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
np.random.seed(4)
from tensorflow import set_random_seed
set_random_seed(4)

def calc_returns(df):
    df['returns'] = df.pct_change()
    df['log-returns'] = np.log(df.iloc[:,0]).diff()
    df['up-down'] = np.sign(df['log-returns'])
    df_dropna = df.dropna()
    return df, df_dropna

def remove_na(df):
    df = df[df['returns'].notna()]
    return df


def get_cwt_features(scale_bot,scale_top,scale_incr,data):
    scales = np.arange(scale_bot,scale_top,step=scale_incr)

    cwt = features.plot_wavelet(time, data, scales)
    # print(type(cwt))
    cwt_features = pd.DataFrame(cwt).T
    cwt_features.set_index(returns.index,inplace=True)
    return cwt_features

def prep_features(data,history_points):
    hist = np.array([data[i:i + history_points].copy() for i in range(len(data) - history_points)])
    return hist

def prep_labels(data,history_points):
    hist_labels = np.array([data[i + history_points].copy() for i in range(len(data) - history_points)])
    hist_labels = np.expand_dims(hist_labels, -1)
    return hist_labels

def split_data(feats, labels, test_split):
    assert feats.shape[0] == labels.shape[0]
    n = int(labels.shape[0]*test_split)
    feature_train = feats[:n]
    label_train = labels[:n]
    feature_test = feats[n:]
    label_test = labels[n:]
    return feature_train, label_train, feature_test, label_test

def test(hist_feats,feature_train,feature_test,label_train,label_test,epoch,batch):
    feat_shape_ax1 = hist_feats.shape[1]
    feat_shape_ax2 = hist_feats.shape[2]
    lstm_input = Input(shape=(feat_shape_ax1, feat_shape_ax2), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
#     output = Activation('sigmoid', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='mse')

    model.fit(x=feature_train, y=label_train, batch_size=batch, epochs=epoch, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate(feature_test, label_test)
    print(evaluation)

    test_predicted = model.predict(feature_test)
    # plt.plot(test_predicted,'o')
    # plt.plot(label_test,'+')
    # plt.legend(['predicted','real'])
    # plt.show()
    return test_predicted, label_test

# not used
def test2(hist_feats,feature_train,feature_test,label_train,label_test,epoch):
    feat_shape_ax1 = hist_feats.shape[1]
    feat_shape_ax2 = hist_feats.shape[2]
    lstm_input = Input(shape=(feat_shape_ax1, feat_shape_ax2), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)

    y = LSTM(50, name='lstm_1')(x)
    y = Dropout(0.2, name='lstm_dropout_1')(y)
    y = Dense(64, name='dense_0')(y)
    y = Activation('sigmoid', name='sigmoid_0')(y)
    y = Dense(1, name='dense_1')(y)

    output = Activation('sigmoid', name='linear_output')(y)
    model = Model(inputs=lstm_input, outputs=output)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='mse')

    model.fit(x=feature_train, y=label_train, batch_size=batch, epochs=epoch, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate(feature_test, label_test)
    print(evaluation)

    test_predicted = model.predict(feature_test)
    # plt.plot(test_predicted,'o')
    # plt.plot(label_test,'+')
    # plt.legend(['predicted','real'])
    # plt.show()
    return test_predicted, label_test

def test_stats(predicted, real):
    c = 0
    s = 0
    for i in range(len(predicted)):
        if (predicted[i] > 0) and (real[i] > 0):
            c = c+1
        if (predicted[i] < 0) and (real[i] < 0):
            c = c+1
        s = s+1
    print('da',c/s)
    pct_correct_da = c/s
    
    return pct_correct_da
# get data
start = '2002-01-01'
end = '2019-01-10'
ticker = 'AAPL'

df = data_reader.download(ticker,start,end)


opens = df['adjusted close'].to_frame()
opens, returns = calc_returns(opens)
print(opens)

# generate log signal
signal = df['adjusted close'].dropna().to_numpy()
log_signal = returns['log-returns'].dropna().to_numpy()


data = log_signal
N = len(data)
t0=0
dt=1/365
time = np.arange(0, N) * dt + t0




lbls = returns['log-returns'].dropna().to_numpy() 

test_split = 0.9


# report for saving testing results
report = pd.DataFrame(columns=['Epoch','Batch Size','CWT Top','CWT Incr','Hist points','Pct Acc'])

# set hyperparameters
batches = [8]
tops = [10]
hist_list = [10]
epochs = [150]
for i in tops:
    for b in batches:
        for k in hist_list:
            for e in epochs:
                
                print("running: hist " + str(k) + ", cwt top "+ str(i) + ",batch "+str(b) )
                scale_bot = 1
                scale_top = i
                scale_incr = 1

                cwt_features = get_cwt_features(scale_bot,scale_top,scale_incr,data)
                results = pd.concat([opens['up-down'],opens['log-returns'],cwt_features],axis=1,sort=False)

                feats = results.dropna().to_numpy()
                history_points = k
                hist_feats = prep_features(feats,history_points)

                hist_labels = prep_labels(lbls,history_points)

                
                feature_train, label_train, feature_test, label_test = split_data(hist_feats,hist_labels,0.9)

                epoch = e
                batch = b 
                predicted, real = test(hist_feats,feature_train,feature_test,label_train,label_test,epoch,batch) 
                pct_da = test_stats(predicted, real) 

                report = report.append({'Epoch':e,'Batch Size':b,'CWT Top':i,'CWT Incr':1,'Hist points':k,'Pct Acc':pct_da},ignore_index=True)
                
                test_datapoints = pd.DataFrame(data={'Pred':predicted.T[0],'Actual':real.T[0]})
                fname_test_dpoints = str(ticker)+'_datapoints_e'+str(e)+'b'+str(b)+'cwttop'+str(i)+'cwtinc'+str(1)+'h'+str(k)+'pctacc'+str(pct_da)+'.csv'
                data_reader.save_df(test_datapoints,fname_test_dpoints)

data_reader.save_df(report,'AAPL_hypertests.csv')