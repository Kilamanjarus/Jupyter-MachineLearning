from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
music_data = pd.read_csv('music.csv')
print(music_data)
X = music_data.drop(columns=['genre'])
y = music_data['genre']
print(X)
print(y)

model = DecisionTreeClassifier()
model.fit(X.values, y)
predictions = model.predict([[21, 1], [22, 0]])
predictions


# Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)


lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print(lr_train_mse)
print(lr_test_mse)
print(lr_train_r2)
print(lr_test_r2)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse,
                          lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE',
                      "Training R2", "Test MSE", "Test R2"]
lr_results
