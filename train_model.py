import seaborn as sns
import matplotlib.pyplot as plt
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load dataset
depression_df = pd.read_csv('reddit_depression_dataset.csv', low_memory=False)

# Filter the relevant columns
depression_df = depression_df[['body', 'label']]

# Display the first few rows
print(depression_df.head())

# Drop NA values
depression_df.dropna(inplace=True)
depression_df.reset_index(drop=True, inplace=True)

# Check for remaining NA values
print(depression_df.isna().sum())

# Extract the same number of records from both classes
number_records = 461744
np.unique(depression_df.label, return_counts=True)

# Extracting the same number of records from both classes
df1 = depression_df.loc[depression_df['label'] == 0][:number_records]
df2 = depression_df.loc[depression_df['label'] == 1][:number_records]

# Creating the final DataFrame
depression_df = pd.concat([df1, df2])

# Checking class distribution
sns.countplot(x=depression_df.label)
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('class_distribution.png')  # Save the plot as a PNG file
plt.show()  # Show the plot

# Preparing data for model
X = depression_df.iloc[:, 0].values
y = depression_df.iloc[:, 1].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=42)

# Model building function
def buildModel(clf, vectorizer):
    model = Pipeline([
        ('vect', vectorizer),
        ('clf', clf)
    ])
    return model

# Build and fit the model
model = buildModel(clf=MultinomialNB(), vectorizer=CountVectorizer())
model.fit(X_train, y_train)

# Evaluate the model
print("Model Score:", model.score(X_test, y_test))

# Predictions and confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the plot as a PNG file
plt.show()  # Show the plot

# Print classification report
print(classification_report(y_test, y_pred))

# After evaluating the model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save the accuracy to a text file
with open('model_accuracy.txt', 'w') as f:
    f.write(f"{accuracy:.2f}")

# Save the model to a file
joblib.dump(model, 'depression_model.pkl')
print("Model saved as 'depression_model.pkl'")

joblib.dump(model.named_steps['vect'], 'vectorizer.pkl')
