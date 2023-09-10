import pandas as pd
import requests
from bs4 import BeautifulSoup

main_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin" # 마지막 회차를 얻기 위한 주소
basic_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin&drwNo=" # 임의의 회차를 얻기 위한 주소

# # 마지막 회차 정보를 가져옴
# def GetLast(): 
#     resp = requests.get(main_url)
#     soup = BeautifulSoup(resp.text, "lxml")
#     result = str(soup.find("meta", {"id" : "desc", "name" : "description"})['content'])
#     s_idx = result.find(" ")
    
#     e_idx = result.find("회")
#     return int(result[s_idx + 1 : e_idx])

# # 지정된 파일에 지정된 범위의 회차 정보를 기록함
# def Crawler(s_count, e_count, fp):
#     for i in range(s_count , e_count + 1):
#         crawler_url = basic_url + str(i)
#         resp = requests.get(crawler_url)
#         soup = BeautifulSoup(resp.text, "html.parser")

#         text = soup.text

#         s_idx = text.find(" 당첨결과")
#         s_idx = text.find("당첨번호", s_idx) + 4
#         e_idx = text.find("보너스", s_idx)
#         numbers = text[s_idx:e_idx].strip().split()

#         s_idx = e_idx + 3
#         e_idx = s_idx + 3
#         bonus = text[s_idx:e_idx].strip()

#         s_idx = text.find("1등", e_idx) + 2
#         e_idx = text.find("원", s_idx) + 1
#         e_idx = text.find("원", e_idx)
#         money1 = text[s_idx:e_idx].strip().replace(',','').split()[2]

#         s_idx = text.find("2등", e_idx) + 2
#         e_idx = text.find("원", s_idx) + 1
#         e_idx = text.find("원", e_idx)
#         money2 = text[s_idx:e_idx].strip().replace(',','').split()[2]

#         s_idx = text.find("3등", e_idx) + 2
#         e_idx = text.find("원", s_idx) + 1
#         e_idx = text.find("원", e_idx)
#         money3 = text[s_idx:e_idx].strip().replace(',','').split()[2]

#         s_idx = text.find("4등", e_idx) + 2
#         e_idx = text.find("원", s_idx) + 1
#         e_idx = text.find("원", e_idx)
#         money4 = text[s_idx:e_idx].strip().replace(',','').split()[2]

#         s_idx = text.find("5등", e_idx) + 2
#         e_idx = text.find("원", s_idx) + 1
#         e_idx = text.find("원", e_idx)
#         money5 = text[s_idx:e_idx].strip().replace(',','').split()[2]

#         line = str(i) + ',' + numbers[0] + ',' + numbers[1] + ',' + numbers[2] + ',' + numbers[3] + ',' + numbers[4] + ',' + numbers[5] + ',' + bonus + ',' + money1 + ',' + money2 + ',' + money3 + ',' + money4 + ',' + money5
#         print(line)
#         line += '\n'
#         fp.write(line)

# last = GetLast() # 마지막 회차를 가져옴

# fp = open('2020-1-25-keras_lstm_lotto_v895_data.csv', 'w')
# Crawler(1, last, fp) # 처음부터 마지막 회차까지 저장
# fp.close()
# print(fp.read())

import numpy as np

# 당첨번호를 원핫인코딩벡터(ohbin)으로 변환
def numbers2ohbin(numbers):

    ohbin = np.zeros(45) #45개의 빈 칸을 만듬

    for i in range(6): #여섯개의 당첨번호에 대해서 반복함
        ohbin[int(numbers[i])-1] = 1 #로또번호가 1부터 시작하지만 벡터의 인덱스 시작은 0부터 시작하므로 1을 뺌
    
    return ohbin

# 원핫인코딩벡터(ohbin)를 번호로 변환
def ohbin2numbers(ohbin):

    numbers = []
    
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0: # 1.0으로 설정되어 있으면 해당 번호를 반환값에 추가한다.
            numbers.append(i+1)
    
    return numbers

print("1:" + str(numbers2ohbin([10,23,29,33,37,40])))
print("2:" + str(numbers2ohbin([9,13,21,25,32,42])))

rows = np.loadtxt("./2020-1-25-keras_lstm_lotto_v895_data.csv", delimiter=",")
row_count = len(rows)

print("row count: " + str(row_count))
numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin, numbers))

x_samples = ohbins[0:row_count-1]
y_samples = ohbins[1:row_count]

#원핫인코딩으로 표시
print("ohbins")
print("X[0]: " + str(x_samples[0]))
print("Y[0]: " + str(y_samples[0]))

#번호로 표시
print("numbers")
print("X[0]: " + str(ohbin2numbers(x_samples[0])))
print("Y[0]: " + str(ohbin2numbers(y_samples[0])))

train_idx = (0, 700)
val_idx = (700, 800)
test_idx = (800, len(x_samples))

print("train: {0}, val: {1}, test: {2}".format(train_idx, val_idx, test_idx))

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

# 모델을 정의합니다.
model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True),
    keras.layers.Dense(45, activation='sigmoid')
])

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 매 에포크마다 훈련과 검증의 손실 및 정확도를 기록하기 위한 변수
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# 최대 100번 에포크까지 수행
for epoch in range(100):

    model.reset_states() # 중요! 매 에포크마다 1회부터 다시 훈련하므로 상태 초기화 필요

    batch_train_loss = []
    batch_train_acc = []
    
    for i in range(train_idx[0], train_idx[1]):
        
        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)
        
        loss, acc = model.train_on_batch(xs, ys) #배치만큼 모델에 학습시킴

        batch_train_loss.append(loss)
        batch_train_acc.append(acc)

    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    batch_val_loss = []
    batch_val_acc = []

    for i in range(val_idx[0], val_idx[1]):

        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)
        
        loss, acc = model.test_on_batch(xs, ys) #배치만큼 모델에 입력하여 나온 답을 정답과 비교함
        
        batch_val_loss.append(loss)
        batch_val_acc.append(acc)

    val_loss.append(np.mean(batch_val_loss))
    val_acc.append(np.mean(batch_val_acc))

    print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f} val acc {3:0.3f} loss {4:0.3f}'.format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss), np.mean(batch_val_acc), np.mean(batch_val_loss)))

    model.save('model_{0:04d}.h5'.format(epoch+1))
