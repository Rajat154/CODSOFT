# CREDIT CARD FRAUD DETECTION - Clean Python Script

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the Datasets
train_file = r"C:\Users\DELL\Desktop\intership project\CODSOFT-main\credit card\fraudTrain.csv"
test_file = r"C:\Users\DELL\Desktop\intership project\CODSOFT-main\credit card\fraudTest.csv"

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Combine Train and Test for Preprocessing
df_combined = pd.concat([df_train, df_test], axis=0)

# Drop Unnecessary Columns
columns_to_drop = ["first", "last", "job", "dob", "trans_num", "street", "trans_date_trans_time", "city", "state"]
df_combined.drop(columns=columns_to_drop, axis=1, inplace=True)

# Visualization: Gender Distribution
sns.countplot(x='gender', data=df_combined)
plt.title("Gender Distribution")
plt.show()

# Visualization: Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = df_combined.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Label Encoding for Categorical Variables
encoder = LabelEncoder()
df_combined["merchant_new"] = encoder.fit_transform(df_combined["merchant"])
df_combined.drop("merchant", axis=1, inplace=True)

df_combined["category_new"] = encoder.fit_transform(df_combined["category"])
df_combined.drop("category", axis=1, inplace=True)

# One-Hot Encoding for Gender
df_combined = pd.get_dummies(df_combined)
df_combined.drop('gender_F', axis=1, inplace=True)

# Split Data into Features and Target
X = df_combined.drop("is_fraud", axis=1)
y = df_combined["is_fraud"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------- Logistic Regression -------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("Accuracy:", round(accuracy_score(y_test, lr_pred), 4))

# ------------------- Decision Tree -------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Results:")
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print("Accuracy:", round(accuracy_score(y_test, dt_pred), 4))

# ------------------- Random Forest -------------------
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))
