import pandas
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score

raw_df = pandas.read_csv("Data/mail_data.csv")

print(raw_df.head())

#Preprocessing

data = raw_df.copy()

data["Category"].replace(["ham", "spam"], [1, 0], inplace = True)

data['Category'].replace('', np.nan, inplace = True)
data['Message'].replace('', np.nan, inplace = True)

data.dropna(inplace = True)

print(data.info())

X = data["Message"]
y = data["Category"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 20)

feature_extraction = TfidfVectorizer()

X_train = feature_extraction.fit_transform(X_train_raw)
X_test = feature_extraction.transform(X_test_raw)

#Training and validation

model = LogisticRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)

print([acc, prec, rec])

#Testing model

input_mail = ["Hello, u won a free phone!"]

input_data = feature_extraction.transform(input_mail)

answer = model.predict(input_data)[0]

dict = {0: "spam", 1: "legit"}

print(dict[answer])