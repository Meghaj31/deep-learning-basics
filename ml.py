import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('DATA/fake_reg.csv')
# print(df.head())
# sns.pairplot(df)
# plt.show()
# sns.boxplot(x='feature1',y='feature2',data=df)
# plt.show()

from sklearn.model_selection import train_test_split
X=df[['feature1','feature2']].values
y=df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(X_train.min())

# creating neural network
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
# model=Sequential([Dense(4,'relu'),
#                   Dense(2,'relu'),
#                   Dense(1)])
model = Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1))
model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train,y=y_train,epochs=250)
# loss_df=pd.DataFrame(model.history.history)
# loss_df.plot()
# plt.show()
print(model.evaluate(X_test,y_test,verbose=0))
test_pred=model.predict(X_test)
print(test_pred)
test_pred=pd.Series(test_pred.reshape(300,))
print(test_pred)
pred_df=pd.DataFrame(y_test,columns=['Test True Y'])
pred_df=pd.concat([pred_df,test_pred],axis=1)
pred_df.columns=['Test True Y','Model Predictions']
# sns.scatterplot(x='Test True Y',y='Model Predictions',data=pred_df)
# plt.show()
from sklearn.metrics import mean_squared_error,mean_absolute_error
# print(mean_absolute_error(pred_df['Test True Y'],pred_df['Model Predictions']))
# print(mean_squared_error(pred_df['Test True Y'],pred_df['Model Predictions']))
# print(df.describe())
new_data=[[239,323]]
new_data=scaler.transform(new_data)
print(model.predict(new_data))