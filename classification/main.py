import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData():
    data = pd.read_excel("WafeForm1.xlsx", header=None)
    return data

def localMax(data):
    funcs = []
    for i in range(len(data.columns(0))):
        funcs.append(data.iloc[i, :])
    print(funcs)

def main():
    data = loadData()
    localMax(data)

if __name__ == '__main__':
    main()