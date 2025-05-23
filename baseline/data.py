import csv
import os.path as osp
import pickle
import numpy as np
import pandas as pd


class GetData(object):

    def __init__(self, data='x', drug_num=1069):
        super().__init__()
        self.drug_num = drug_num
        self.S1, self.S2, self.X = self.__get__ADMM__data__(data)

    def __get__ADMM__data__(self, data):
        s1_type_name = [
            "cdcdb_tdc_si_0",
            "cdcdb_tdc_si_1"
        ]
        s2_type_name = [
            "cdcdb_tdc_si_2",
            "cdcdb_tdc_si_3"
        ]

        s1_all = []
        s2_all = []

        for name in s1_type_name:
            df = pd.read_csv("ADMM/" + name + ".csv", index_col=0)
            mat = df.values
            s1_all.append(mat)

        for name in s2_type_name:
            df = pd.read_csv("ADMM/" + name + ".csv", index_col=0)
            mat = df.values
            s2_all.append(mat)

        s1 = np.mean(np.stack(s1_all), axis=0)
        s2 = np.mean(np.stack(s2_all), axis=0)

        tensor_name = f'cdcdb_tdc_{data}.pickle'

        with open("ADMM/" + tensor_name, 'rb') as t_x:
            x_pickle = pickle.load(t_x)

        x_keys = sorted(x_pickle.keys(), key=lambda x: int(x.split()[0]))
        tensor_slices_x = [x_pickle[key].values for key in x_keys]
        for matrix in tensor_slices_x:
            if not np.allclose(matrix, matrix.T):
                raise ValueError("X matrix not symmetric")
        x = np.stack(tensor_slices_x, axis=-1)

        print(f"tensor shape: {x.shape}")

        return s1, s2, x
