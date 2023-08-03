import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error




path = "bsa.csv"

df = pd.read_csv(path,  low_memory=False)

df2=df.dropna()



X = df2[["barrio","ambientes","metro_cuadrado"]]


X = pd.get_dummies(data=X, drop_first=True)         #agrego dummies para los barrios al data frame asi poder usarlos en el modelo
Y = df2[["precio"]]
inputs = (X.head(1).copy())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)   #separo el df para unos entrenarlo y otro testearlo


regr = linear_model.LinearRegression()

regr.fit (X_train, Y_train)             #llamada para entrenar al modelo
print ('Coeficientes:', regr.coef_)   #coeficientes tita





y_sombrero = regr.predict(X_test)    #llamada para hacer la prediccion
print(y_sombrero)



print("MSE:%.2f"  % mean_squared_error(Y_test,y_sombrero)) #calculo el MSE para ver cuanto error residual hay

print('Variance score: %.2f' % regr.score(X_test, Y_test))  #mientras mas cerca de1 mas cercana es la prediccion


inputs.iloc[0,1] = 0
inputs.iloc[0,0] = 0
inputs["barrio_Agronomía"] = False



cantambiente = input("cantidad de ambientes: ")
cantmetros = input("cantidad de metros cuadrados: ")   
barrio = input("barrio: ")   
str = "barrio_" + barrio
inputs["ambientes"] = int(cantambiente)
inputs["metro_cuadrado"] = int(cantmetros)
inputs[str] = True

a = np.asanyarray(inputs)
y_sombrero_input = regr.predict(inputs) 

print("precio aproximado:",  int (y_sombrero_input[0][0]))