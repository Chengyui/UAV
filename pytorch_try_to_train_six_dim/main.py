import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# 生成标签值：下一天收盘价（涉及删除最后一条数据，不要重复执行该函数）
def generate_label(data_path):
    df = pd.read_csv(data_path)

    for i in ['xa','ya','za','row','pitch','yaw']:
        next_close = list()
        for j in range(len(df[i])-1):
            next_close.append(df[i][j+1])
        next_close.append(0)
        df['new'+i] = next_close
    df.to_csv('new_UVA.csv',index=None)

# 生成训练和测试数据
def generate_model_data(data_path, alpha, seq_len):
    df = pd.read_csv(data_path)
    train_seq = int((len(df) - seq_len + 1))
    for property in ['xa','ya','za','row','pitch','yaw','newxa','newya','newza','newrow','newpitch','newyaw']:
        df[property] = scaler.fit_transform(np.reshape(np.array(df[property]), (-1, 1)))

    X_data, Y_data = list(), list()
    # 生成时序数据
    for i in range(train_seq):
        for k in ['newxa','newya','newza','newrow','newpitch','newyaw']:
            Y_data.append(df[k][i + seq_len - 1])
        for j in range(seq_len):
            for m in ['xa','ya','za','row','pitch','yaw']:
                X_data.append(df[m][i + j])
    X_data = np.reshape(np.array(X_data), (-1, 6 * seq_len))  # 5表示特征数量*天数
    train_length = int(len(Y_data) * alpha)
    X_train = np.reshape(np.array(X_data[:train_length]), (len(X_data[:train_length]), seq_len, 6))
    X_test = np.reshape(np.array(X_data[train_length:]), (len(X_data[train_length:]), seq_len, 6))
    #Y_train, Y_test = np.array(Y_data[:train_length]), np.array(Y_data[train_length:])
    Y_train = np.reshape(np.array(Y_data[:train_length]),(len(Y_data[:train_length]),  1, 6))
    Y_test = np.reshape(np.array(Y_data[train_length:]), (len(Y_data[train_length:]), 1, 6))
    return X_train, Y_train, X_test, Y_test


def calc_MAPE(real, predict):
    Score_MAPE = 0
    for i in range(len(predict[:, 0])):
        Score_MAPE += abs((predict[:, 0][i] - real[:, 0][i]) / real[:, 0][i])
    Score_MAPE = Score_MAPE * 100 / len(predict[:, 0])
    return Score_MAPE


def calc_AMAPE(real, predict):
    Score_AMAPE = 0
    Score_MAPE_DIV = sum(real[:, 0]) / len(real[:, 0])
    for i in range(len(predict[:, 0])):
        Score_AMAPE += abs((predict[:, 0][i] - real[:, 0][i]) / Score_MAPE_DIV)
    Score_AMAPE = Score_AMAPE * 100 / len(predict[:, 0])
    return Score_AMAPE


def evaluate(real, predict):
    RMSE = math.sqrt(mean_squared_error(real[:, 0], predict[:, 0]))
    MAE = mean_absolute_error(real[:, 0], predict[:, 0])
    MAPE = calc_MAPE(real, predict)
    AMAPE = calc_AMAPE(real, predict)
    return RMSE, MAE, MAPE, AMAPE


def lstm_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(units=20, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=200, batch_size=20, verbose=1)

    trainPredict = model.predict(X_train)
    trainPredict = scaler.inverse_transform(trainPredict)
    Y_train = scaler.inverse_transform(np.reshape(Y_train, (-1, 1)))

    testPredict = model.predict(X_test)
    testPredict = scaler.inverse_transform(testPredict)
    Y_test = scaler.inverse_transform(np.reshape(Y_test, (-1, 1)))

    return Y_train, trainPredict, Y_test, testPredict


if __name__ == '__main__':
    data_path = 'UAV.csv'
    seq_len = 15
    alpha = 0.8
    #generate_label(data_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, Y_train, X_test, Y_test = generate_model_data('new_UAV.csv', alpha, seq_len)
    train_Y, trainPredict, test_Y, testPredict = lstm_model(X_train, Y_train, X_test, Y_test)

    RMSE, MAE, MAPE, AMAPE = evaluate(test_Y, testPredict)
    print(RMSE, MAE, MAPE, AMAPE)
    plt.subplot(121)
    plt.plot(list(trainPredict), color='red', label='prediction')
    plt.plot(list(train_Y), color='blue', label='real')
    plt.legend(loc='upper left')
    plt.title('train data')

    plt.subplot(122)
    plt.plot(list(testPredict), color='red', label='prediction')
    plt.plot(list(test_Y), color='blue', label='real')
    plt.legend(loc='upper left')
    plt.title('test data')
    plt.suptitle('units==20,RMSE=\d,MAE=\d,MAPE=\d,AMAPE=\d',RMSE,MAE,MAPE,AMAPE)
    plt.show()


