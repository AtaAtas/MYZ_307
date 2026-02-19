import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

"""
Kemal Yağız Daşkıran Bahar 2025 

Gauss.py'dan elde edilen csv dosyalarını okuyarak bu değerlerden Confusion Matrix, Precision, Recall ve F1 değerlerini üretir.

"""

df1 = pd.read_csv("./q_0_class.csv", header=None)
df2 = pd.read_csv("./q_1_class.csv", header=None)
data1 = df1.values.flatten()
data2 = df2.values.flatten()

def ortalama(data):
    return sum(data)/len(data)

def standart_sapma(data):
    ort = ortalama(data)
    x = 0
    for item in data:
        x += (item - ort)**2
    x /= (len(data)-1)
    return sqrt(x)

#Numpy ve kendi fonksiyonlarımın kullanılarak ortalamanın elde edilmesi ve printlenmesi
print(f"\n\n(Numpy Versiyon) 1 sınıflı verilerin ortalama değeri: {np.average(data2)}\n0 sınıflı verilerin ortalama değeri: {np.average(data1)}")
print(f"\n(Ben Versiyon) Oluşturulan 1 sınıflı verilerin ortalama değeri: {ortalama(data2)}\n0 sınıflı verilerin ortalama değeri: {ortalama(data1)}")

#Numpy ve kendi fonksiyonlarımın kullanılarak standart sapmanın elde edilmesi ve printlenmesi
print(f"\n\n(Numpy Versiyon) 1 sınıflı verilerin standart sapma değeri: {np.std(data2)}\n0 sınıflı verilerin standart sapma değeri: {np.std(data1)}")
print(f"\n(Ben Versiyon) Oluşturulan 1 sınıflı verilerin standart sapma değeri: {standart_sapma(data2)}\n0 sınıflı verilerin standart sapma değeri: {standart_sapma(data1)}")

#Confusion Matrix'i oluşturmak için gerekli değişkenlerin tanımı
tp,fp,fn,tn = 0,0,0,0
esik_degeri = 1

# 0 sınıflı veilerde Eşik değerinden düşük ve eşit değerler True Negative olarak toplanıyor. Aksiler ise False Negative olarak.
tn = sum(1 for item in data1 if item <= esik_degeri)
fn = sum(1 for item in data1 if item > esik_degeri)

# 1 sınıflı verilerde 1 eşik değerinin üstünde ve eşit değerler True Positive olarak değerlendirilirken aksi olanlar False Positive olarak değerlendiriliyor.
tp = sum(1 for item in data2 if item >= esik_degeri)
fp = sum(1 for item in data2 if item < esik_degeri)

print(f"\nTrue Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}\n")

confusion,confusion_list = ["True Positive", "False Positive", "False Negative", "True Negative"],[tp,fp,fn,tn]

plt.figure()
plt.bar(confusion,confusion_list)
plt.grid(True)
plt.savefig("./bar_confusion_plot.png")


precision_1, recall_1 = tp/(tp+fp), tp/(tp+fn)
f1_score_1 = 2*(precision_1*recall_1)/(precision_1+recall_1)

precision_0,recall_0 = tn/(tn+fn), tn/(tn+fp)
f1_score_0 = 2*(precision_0*recall_0)/(precision_0+recall_0)

print(f"For Class 1:\nPrecision: {precision_1}\nRecall:{recall_1}\nF1 Score: {f1_score_1}\n\n")
print(f"For Class 0:\nPrecision: {precision_0}\nRecall:{recall_0}\nF1 Score: {f1_score_0}")

