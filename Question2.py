import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


class DataLoader():

    def __init__(self):
        pass

    @staticmethod
    def read_data_(file_path_, file_name_):
        assert file_name_.split('.')[-1] == "csv", 'Invalid file format.'

        df = pd.read_csv(file_path_ + "\\" + file_name_)
        # print(type(df.ix[:, 'St':'Vt1']))
        # print df.ix[1, 'St':'Vt1']

        label = df.ix[:, 'St'].values
        data  = df.ix[:, 'EMAt1': 'CMFt2'].values

        return data, label


if __name__ == "__main__":

    data_path_ = "./deepae_Dataset(tr,te)"
    dt = DataLoader()

    svr_dict = {"MAE"    : [],
                "RMSE"   : [],
                "CVRMES" : []}

    dnn_dict = {"MAE"    : [],
                "RMSE"   : [],
                "CVRMES" : []}
    for i in range(1,11):
        tr_file_name_ = "cv%d_tr.csv" %i
        te_file_name_ = "cv%d_te.csv" %i

        train_data , train_label = dt.read_data_(data_path_, tr_file_name_)
        test_data, test_label = dt.read_data_(data_path_, te_file_name_)

        #############################################
        #                  train                    #
        #############################################
        svr = SVR().fit(train_data, train_label)
        dnn = MLPRegressor().fit(train_data, train_label)

        #############################################
        #                  predict                  #
        #############################################
        svr_result = svr.predict(test_data)
        dnn_result = dnn.predict(test_data)

        n = len(test_label)
        svr_disparity = test_label - svr_result
        dnn_disparity = test_label - dnn_result

        svr_MAE = sum(map(abs, svr_disparity)) / n
        dnn_MAE = sum(map(abs, dnn_disparity)) / n
        svr_dict["MAE"].append(svr_MAE)
        dnn_dict["MAE"].append(dnn_MAE)

        svr_RMSE = np.sqrt(sum(map(pow, svr_disparity)) / n)
        dnn_RMSE = np.sqrt(sum(map(pow, dnn_disparity)) / n)
        svr_dict["RMSE"].append(svr_RMSE)
        dnn_dict["RMSE"].append(dnn_RMSE)

        svr_CVRMSE = svr_RMSE / sum(test_label) / n
        dnn_CVRMSE = dnn_RMSE / sum(test_label) / n
        svr_dict["CVRMSE"].append(svr_CVRMSE)
        dnn_dict["CVRMSE"].append(dnn_CVRMSE)



