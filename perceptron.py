import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
#import random


da0 = pd.read_csv('data0.csv')#手書き文字0のデータの読み込み
da1 = pd.read_csv("data1.csv")#手書き文字1のデータの読み込み

dat0 = da0.values#読み込んだ0のデータを配列で保存
dat1 = da1.values#読み込んだ1のデータを配列で保存
w = np.zeros(256)#重みの初期値
result = np.zeros(300)#結果の初期値

for i in range(0,256):
   # w[i]=np.random.uniform(-0.5, 0.5)#重みを乱数にする
    w[i]=np.random.uniform(-0.1, 0.1)#重みを乱数にする

print(w)

#学習
epoch = 0
alpha=0.0001#学習係数0.0001
for epoch in range(0,300):
    for k in range(0,120):#学習用120枚ずつ
        sum = 0#重み(転置)と入力の総和
        for i in range(0,256):
            sum = sum+(dat0[k][i] * w[i])#0のときtn=-1
        if(sum<=0):#入力の総和が閾値より小さい時
            pass#何もしない
        elif(sum>0):#出力が誤りのときC-
            for j in range(0,256):
                    w[j] = w[j] - (dat0[k][j]*alpha)#wの推定値にxnを引く
        sum = 0#重み(転置)と入力の総和
        for i in range(0,256):
            sum = sum+(dat1[k][i] * w[i])
        if(sum>=0):#出力が正しい答えのとき
            pass#何もしない
        elif(sum<0):#出力が誤りのときC+
            for j in range(0,256):
                    w[j] = w[j] + (dat1[k][j]*alpha)#wの推定値にxnを足す
    


#検証
    true_cnt = 0#正解した個数のカウント
    for check in range(120,159):#テストデータ39*2=78
        sum = 0
        for i in range(0,256):

            sum = sum+(dat0[check][i] * w[i])
        if(sum<=0):
            true_cnt = true_cnt +1
        else:
            true_cnt  = true_cnt  +0
            
        sum = 0
        for i in range(0,256):

            sum = sum+(dat1[check][i] * w[i])
        if(sum>0):
            true_cnt  = true_cnt +1
        else:
            true_cnt  = true_cnt  + 0
        probability = true_cnt/78#正解率の計算
    result[epoch] = probability
    print("epoch :" ,epoch , "正解率　:" ,probability )#結果の出力

print(w)
fig, ax = plt.subplots() 
ax.set_xlabel('epoch')
ax.set_ylabel('probability')
ax.plot(result)
ax.legend()

file = open('result.csv', 'w')
w = csv.writer(file)
w.writerow(result)
