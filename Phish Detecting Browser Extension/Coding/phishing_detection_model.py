import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('phishing_data.csv')  # Load your dataset
X = data['url']  # Features (text data)
y = data['label']  # Labels (target variable)

# Feature extraction
vectorizer = TfidfVectorizer()  # Initialize TF-IDF Vectorizer
X_vectorized = vectorizer.fit_transform(X)  # Convert text data to TF-IDF features

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier()  # Initialize Random Forest Classifier
model.fit(X_train, y_train)  # Train the model

# Make predictions and evaluate the model
y_pred = model.predict(X_test)  # Predict on the test set
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy
