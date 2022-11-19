import pandas as pd
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
print(dataset)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c = []
for i in range(0, 1000):
    reviews = re.sub('[^a-zA-Z]', ' ', dataset["Review"][i])
    reviews = reviews.lower()
    reviews = reviews.split()
    ps = PorterStemmer()
    all_words = stopwords.words("English")
    all_words.remove("not")
    reviews = [ps.stem(word) for word in reviews if word not in set(all_words)]
    reviews = " ".join(reviews)
    c.append(reviews)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(c).toarray()
y = dataset["Liked"]

from sklearn.model_selection import train_test_split as t
x_train, x_test, y_train, y_test = t(x, y, test_size=0.2, random_state=0)

# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(x_train, y_train)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=10)
# model.fit(x_train, y_train)

from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(x_train, y_train)


pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, pred))
cm = confusion_matrix(y_test, pred)
print(cm)