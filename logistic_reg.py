# Import the required libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

# Load csv dataset
file_path = './email.csv'
df = pd.read_csv(file_path)

# Visualize the distribution of message categories
plt.figure(figsize=(6,6))
df['Category'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Ham & Spam messages')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Generation of word clouds for spam and ham messages
spam_words = ' '.join(df[df['Category'] == 'spam']['Message'])
ham_words = ' '.join(df[df['Category'] == 'ham']['Message'])

# Visualizing Data
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=300, height=300, background_color='white').generate(spam_words))
plt.title('Spam messages Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=300, height=300, background_color='white').generate(ham_words))
plt.title('Ham messages Word Cloud')
plt.axis('off')
plt.show()

# Text processing and classification (Text Vectorization)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Message'])
y = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Spam Detection Accuracy: {accuracy:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
