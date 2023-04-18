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
