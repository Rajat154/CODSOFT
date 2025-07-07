# SPAM SMS DETECTION - VS Code Friendly Version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# Set Plot Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load Dataset
file_path = r"C:\Users\DELL\Desktop\intership project\CODSOFT-main\spam\spam.csv"   # Place your CSV in 'data'        

try:
    df = pd.read_csv(file_path, encoding='latin-1', usecols=['v1', 'v2'])
    df.columns = ['Label', 'Message']
except FileNotFoundError:
    print(f"\n❌ File not found at {file_path}. Please check the path.")
    exit()

# Data Preprocessing
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# Plot Class Distribution
sns.countplot(x='Label', data=df, palette='coolwarm')
plt.title('Distribution of Ham and Spam Messages', fontsize=16)
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.show()

# Feature Extraction
X = df['Message']
y = df['Label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Define Models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(kernel='linear', probability=True)
}

# Evaluation Function
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probability for ROC
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n\033[1mModel: {name}\033[0m")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', color=np.random.rand(3,))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.show()

# Evaluate All Models
for model_name, model in models.items():
    evaluate_model(model_name, model)

print("\n All models evaluated successfully!")
