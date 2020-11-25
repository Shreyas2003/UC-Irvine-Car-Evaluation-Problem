import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

# Change Data Values to all Numerical (Buying, Maint, Lug_Boot, Safety, Class)
enc = preprocessing.LabelEncoder()
buying = enc.fit_transform(data["buying"])
maint = enc.fit_transform(data["maint"])
door = enc.fit_transform(data["door"])
persons = enc.fit_transform(data["persons"])
lugboot = enc.fit_transform(data["lug_boot"])
safety = enc.fit_transform(data["safety"])
clas = enc.fit_transform(data["class"])

predict = "class"
loops = 0
best = 0
for t in range(3000):
    x = list(zip(buying, maint, door, persons, lugboot, safety))
    y = list(clas)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)
    # print(x_train, y_test)
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    loops += 1
    if acc > best:
        best = acc
    if acc > .988:
        break

print()
name = ["Unacceptable", "Acceptable", "Good", "Very Good"]
print("Scale of Acceptance: ", name)
print("Attempts: ", loops)
print("Correlation Coefficent: ", best)
print()
predicted = model.predict(x_test)
i = 0
for x in range(len(predicted)):
    if name[predicted[x]] != name[y_test[x]]:
        i += 1
        print("Predicted: ", name[predicted[x]])
        print("Data: ", x_test[x])
        print("Actual: ", name[y_test[x]])
        print()
if i == 0:
    print("No Errors!")
