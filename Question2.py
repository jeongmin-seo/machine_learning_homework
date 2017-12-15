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


def normalize(data_, label_, norm_relate_):

    data = (data_ - norm_relate_.ix['min', 'EMAt1': 'CMFt2']) / \
           (norm_relate_.ix['max', 'EMAt1': 'CMFt2'] - norm_relate_.ix['min', 'EMAt1': 'CMFt2'])

    label = (label_ - norm_relate_.ix['min', 'St']) / \
            (norm_relate_['max', 'EMAt1': 'CMFt2'] - norm_relate_['min', 'EMAt1': 'CMFt2'])

    return data, label


def de_normalize(norm_label_, norm_relate_):

    return norm_label_ * (norm_relate_.ix['max', 'St'] - norm_relate_.ix['min', 'St']) + norm_relate_.ix['min', 'St']

if __name__ == "__main__":

    norm_path_ = "./KospiData.xlsx"
    min_max_ = pd.read_excel(norm_path_)

    data_path_ = "./deepae_Dataset(tr,te)"
    dt = DataLoader()

    svr_dict = {"MAE"    : [],
                "RMSE"   : [],
                "CVRMES" : []}

    dnn_dict = {"MAE"    : [],
                "RMSE"   : [],
                "CVRMES" : []}

    for i in range(1, 11):
        tr_file_name_ = "cv%d_tr.csv" % i
        te_file_name_ = "cv%d_te.csv" % i

        train_data, train_label = dt.read_data_(data_path_, tr_file_name_)
        test_data, test_label = dt.read_data_(data_path_, te_file_name_)

        #############################################
        #                 normalize                 #
        #############################################
        norm_train_data, norm_train_label = normalize(train_data, train_label, min_max_)
        norm_test_data, norm_test_label = normalize(test_data, test_data, min_max_)

        #############################################
        #                  train                    #
        #############################################
        svr = SVR().fit(norm_train_data, norm_train_label)
        dnn = MLPRegressor().fit(norm_train_data, norm_train_label)

        #############################################
        #                  predict                  #
        #############################################
        svr_result = svr.predict(norm_test_data)
        dnn_result = dnn.predict(norm_test_data)

        #############################################
        #            de-normalize result            #
        #############################################
        de_norm_svr_result = de_normalize(svr_result, min_max_)
        de_norm_dnn_result = de_normalize(dnn_result, min_max_)

        n = len(test_label)
        svr_disparity = test_label - de_norm_svr_result
        dnn_disparity = test_label - de_norm_dnn_result

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



