import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CRF_Record_PATH='./c1.csv'

def main():
    # f=open(CRF_Record_PATH,'r')
    # csv_reader = csv.reader(f)
    # csv_writer.writerow(["algorithm", "c1", "c2","max_iterations","precision","recall","f1_score"])
    # data = pd.read_csv(CRF_Record_PATH) #读取文件中所有数据
    #
    # 按列分离数据  algorithm,c1,c2,max_iterations,precision,recall,f1_score
    # x = data['']  # 读取某两列
    x=[32,64,128,256]
    y1= [0.9540,0.9553,0.9516,0.9516]
    y2= [0.9539,0.9549,0.9515,0.9513]
    y3 = [0.9536,0.9547,0.9513,0.9511]

    print(x)
    print(type(x))

    plt.plot(x, y1,'red')
    plt.plot(x, y2, 'green')
    plt.plot(x,y3,'blue')
    # print("start drawing")
    plt.xlabel("batch_size")
    plt.ylabel("value")
    plt.title("lr=0.001  red:precision  green:recall  blue:F1_score")
    plt.show()

if __name__ == "__main__":
    main()