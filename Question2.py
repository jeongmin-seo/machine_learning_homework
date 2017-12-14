import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataLoader():

    def __init__(self):
        pass

    def read_data_(self, file_path_):
        df = pd.read_csv(file_path_)
        print(df.ix[:, 'St':'Vt1'])



if __name__=="__main__":

    dt = DataLoader()
    dt.read_data_("D:\\workspace\\github\\machine_learning_homework\\deepae_Dataset(tr,te)\\cv1_te.csv")