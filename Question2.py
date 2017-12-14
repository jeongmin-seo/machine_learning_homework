import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class DataLoader():

    def __init__(self):
        pass

    def read_data_(self, file_path_, file_name_):
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

    for i in range(0,10):
        tr_file_name_ = "cv%d_tr.csv" %i
        te_file_name_ = "cv%d_te.csv" %i

        train_data , train_label = dt.read_data_(data_path_, tr_file_name_)
        test_data, test_label = dt.read_data_(data_path_, te_file_name_)



