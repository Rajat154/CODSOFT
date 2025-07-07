import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score

# Sample Data
train_data = {
    'SN': [1, 2, 3, 4, 5],
    'movie_name': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'category': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
    'confession': [
        'Explosions and car chases everywhere!',
        'Hilarious dialogues and funny moments.',
        'Emotional storyline and strong acting.',
        'Fast-paced thrilling scenes.',
        'Laugh out loud jokes and humor.'
    ]
}

test_data = {
    'SN': [6, 7],
    'movie_name': ['Movie F', 'Movie G'],
    'confession': [
        'A serious plot with deep emotions.',
        'High energy stunts and action-packed sequences.'
    ]
}

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# Combine Train and Test Data
df_combined = pd.concat([df_train, df_test], axis=0)

# Visualization 1: Countplot
plt.figure(figsize=(8, 5))
sns.countplot(x='category', data=df_train)
plt.title('Movie Genre Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualization 2: KDE Plot
sns.displot(df_train['category'], kde=True, height=5, aspect=1.2)
plt.xticks(rotation=45)
plt.show()

# Visualization 3: Histogram without KDE
sns.displot(df_train['category'], kde=False, height=5, aspect=1.2)
plt.xticks(rotation=45)
plt.show()

# Visualization 4: Horizontal Bar Chart
plt.figure(figsize=(8, 5))
genre_count = df_train['category'].value_counts()
sns.barplot(x=genre_count.values, y=genre_count.index, orient='h')
plt.xlabel('Count')
plt.ylabel('Genres')
plt.title('Movie Genre Counts')
plt.show()

# Data Preprocessing
le = LabelEncoder()
df_combined['category'] = le.fit_transform(df_combined['category'].astype(str))
df_combined['movie_name'] = le.fit_transform(df_combined['movie_name'].astype(str))

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df_combined['confession'])
y = df_combined['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Model Results:")
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))
print("Accuracy:", accuracy_score(y_test, nb_pred))
print("R2 Score:", r2_score(y_test, nb_pred))

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Model Results:")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))
