

# Phish Detect-Real Time Phish Detecting Browser Extension

This project implements a machine learning model for detecting phishing URLs. The model uses text feature extraction with TF-IDF and classification with a Random Forest algorithm to identify potentially harmful URLs.

## Overview

The code performs the following steps:
1. Loads a dataset containing URLs labeled as phishing or non-phishing.
2. Extracts features from URLs using TF-IDF (Term Frequency-Inverse Document Frequency).
3. Splits the dataset into training and testing sets.
4. Trains a Random Forest classifier using the training set.
5. Evaluates the model's performance on the test set and prints the accuracy score.

## Prerequisites

To run this project, you need Python 3.x and the following Python libraries:
- `pandas`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn
```

## Dataset

The dataset is assumed to be a CSV file named `phishing_data.csv` with the following columns:
- `url`: The URL to be analyzed.
- `label`: The label indicating if the URL is phishing (`1`) or not (`0`).

## File Structure

```
phishing_detection_model.py
phishing_data.csv
```

## Usage

1. **Prepare the Dataset**: Ensure that `phishing_data.csv` is located in the same directory as the script.

2. **Run the Script**:
   
   ```bash
   python phishing_detection_model.py
   ```

3. **Output**: The script will output the accuracy of the model on the test dataset.

## Code Explanation

1. **Loading the Dataset**:
   ```python
   data = pd.read_csv('phishing_data.csv')
   X = data['url']
   y = data['label']
   ```

2. **Feature Extraction**:
   ```python
   vectorizer = TfidfVectorizer()
   X_vectorized = vectorizer.fit_transform(X)
   ```

3. **Data Splitting**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)
   ```

4. **Model Training**:
   ```python
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```

5. **Model Evaluation**:
   ```python
   y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact Information

For any inquiries or contributions, please contact:

**Name**: DEVI SUBADRA VENKATESAN 
**Email**: devisv25@gmail.com  
**Phone**: (602) 921-5626  
**LinkedIn**: [DEVI SUBADRA VENKATESAN](https://www.linkedin.com/in/devisubadravenkatesan)

## Acknowledgments

- Inspired by the need for effective phishing detection systems.
- Utilizes the `scikit-learn` library for machine learning tasks.
