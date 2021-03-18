import pandas as pd
import plotly.express as px
import plotly.graph_objects as gp
import csv
import plotly.figure_factory as pf
import random

import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sb


df = pd.read_csv("P117data.csv")
#print(df)
scores = df[["variance","skewness","curtosis","entropy"]]
print(scores)

results = df["class"]

score_train, score_test, results_train, results_test = tts(scores, results, test_size= 0.25, random_state = 0)




from sklearn.preprocessing import StandardScaler

ss_x = StandardScaler()
score_train = ss_x.fit_transform(score_train)
score_testt = ss_x.transform(score_test)


model = LogisticRegression(random_state = 0)
model.fit(score_train, results_train)


pred = model.predict(score_test)
predict_value_1 = []

for i in pred:
    if i == 0:
        predict_value_1.append("Autherized")
    else:
        predict_value_1.append("Forged")

actual_value_1 = []

for i in results_test.ravel():
    if i == 0:
        actual_value_1.append("No")
    else:
         actual_value_1.append("Yes")

            
labels = ["Authorized", "Forged"]
cm = confusion_matrix(actual_value_1, predict_value_1, labels)
ax = plt.subplot()
sb.heatmap(cm, annot = True, ax = ax)
