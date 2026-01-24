import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData():
    data = pd.read_excel("classification/WaveForm1.xlsx", header=None)
    return data

def localMax(data):
    # print(data)
    funcs = []
    # print(data.iloc[0, :])
    for i in range(len(data.iloc[:, 0])):
        funcs.append(data.iloc[i, :])
    # print(len(funcs))
    localMax = []
    for i in range(len(funcs)):
        localMaxOne = []
        for j in range(1, len(funcs[i]) - 1):
            if((funcs[i][j] > funcs[i][j-1]) and (funcs[i][j] > funcs[i][j+1])):
                localMaxOne.append(j)
        localMax.append(localMaxOne)
        
    return localMax

def calculateAns(localMax):
    ans = 0
    for i in range(len(localMax)):
        if(len(localMax[i]) == 1):
            ans+=1
    return ans

def main():
    data = loadData()
    localMaxs = localMax(data)
    answer = calculateAns(localMaxs)
    print(answer)

if __name__ == '__main__':
    main()