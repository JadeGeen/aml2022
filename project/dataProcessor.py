import pdr
import mlp
import pandas as pd
import numpy as np
import os

class DataProcessor():
    def __init__(self) -> None:
        '''attention: all data should name in right format!'''
        self.path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
        self.acc_file = os.path.join(self.path,"Accelerometer.csv")
        self.bar_file = os.path.join(self.path,"Barometer.csv")    #加速度计
        self.gyr_file = os.path.join(self.path,"Gyroscope.csv")    #气压计
        self.lin_file = os.path.join(self.path,"Linear Accelerometer.csv") #TODO:check whether blank ok!陀螺仪
        self.mag_file = os.path.join(self.path,"Magnetometer.csv")   #磁力
        self.inp_file = os.path.join(self.path,"Location_input.csv")   #位置输入location_input

        self.data = {"acc":pd.read_csv(self.acc_file), \
                    "bar": pd.read_csv(self.bar_file),\
                    "gyr": pd.read_csv(self.gyr_file),\
                    "lin": pd.read_csv(self.lin_file),\
                    "mag": pd.read_csv(self.mag_file),\
                    "inp": pd.read_csv(self.inp_file),\
                    }
        #for pdr
        linear =(self.data["lin"].values)[1:,1:].astype(np.float64)
        gravity = (self.data["acc"].values)[1:,1:].astype(np.float64)
        magnetometer = (self.data["mag"].values)[1:,1:].astype(np.float64)
        gyroscope = (self.data["gyr"].values)[1:,1:].astype(np.float64)
        location_input = (self.data["inp"].values)[1:,[1,2,5]].astype(np.float64) #TODO: write in more stable way
        time = (self.data["mag"].values)[1:,0].astype(np.float64)

        #计算开始是NAN的行数
        self.start_nan = len(self.data["inp"].dropna(axis=0,how="any"))  #-1是减header,-1是index!
        self.start_nan = min(len(self.data["inp"])//10 , self.start_nan)

        self.pdr_model = pdr.Model(li=linear,gr=gravity,ma=magnetometer, gy=gyroscope, \
        input=location_input,time=time)

        


    def get_pdr_model(self)->pdr.Model:
        '''return the pdr_model'''
        return self.pdr_model

    def get_pdr_predict(self):
        '''use pdr to predict'''
        self.pdr_ans = self.data["inp"].copy()
        predict_time = self.data["inp"].values[self.start_nan:,0]
        p_x,p_y,p_a = self.pdr_model.predict_position(self.start_nan,predict_time)  #TODO这里索引早了，但问题不大
        for i in range(self.start_nan,len(self.pdr_ans)):
            self.pdr_ans.iloc[i,1] = p_x[i-self.start_nan]
            self.pdr_ans.iloc[i,2] = p_y[i-self.start_nan]
            self.pdr_ans.iloc[i,5] = p_a[i-self.start_nan]


    def save_csv(self,file = ""):
        '''input the data, save as csv'''
        save_path = os.path.abspath(os.path.join(self.path,file))
        self.pdr_ans.to_csv(save_path,index=None)

