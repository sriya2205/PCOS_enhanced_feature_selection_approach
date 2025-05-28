import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load the input CSV file
main_data = pd.read_csv('ss.csv')  # Replace with your main dataset file path

# Drop rows with missing target values
main_data.dropna(subset=['PCOS (Y/N)'], inplace=True)

# Convert 'PCOS (Y/N)' to numeric if needed
y = main_data['PCOS (Y/N)'].astype(int)

# Define the feature set (X) by dropping non-numeric and target columns
X = main_data.drop(['PCOS (Y/N)'], axis=1)

# Convert any non-numeric columns to numeric using label encoding
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Check if there are any NaN values and fill or remove them
X.fillna(X.mean(), inplace=True)  # Fill NaNs with column means

# Feature scaling (recommended for SVM and KNN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the machine learning models
models = {
    'SVM': SVC(probability=True),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Perceptron': Perceptron()
}

# Dictionary to store the results
results = {}

# Set up plot for ROC curve
plt.figure(figsize=(10, 8))

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert Linear Regression predictions to binary
    if model_name == 'Linear Regression':
        y_pred_binary = np.round(y_pred)  # Convert predictions to binary
    else:
        y_pred_binary = y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)

    # Handle ROC curve generation
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)  # Decision function for classifiers like SVM
    else:
        y_proba = y_pred  # For models without decision_function or predict_proba, use predicted values

    # AUC calculation
    if len(set(y_test)) > 1:  # Check for at least two classes before AUC
        auc_score = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    else:
        auc_score = 0.0

    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC Score': auc_score
    }

# Create a DataFrame for displaying the results
results_df = pd.DataFrame(results).T

# Formatting the output for better presentation (percentage format)
for metric in results_df.columns:
    results_df[metric] = results_df[metric].apply(lambda x: f"{x:.3f} ({x * 100:.1f}%)" if x is not None else "0.000 (0.0%)")

# Display the results in a table format
print("Model Evaluation Metrics:")
print(results_df.to_string(index=True, header=True))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Finding the model with the highest metrics
highest_accuracy_model = results_df['Accuracy'].idxmax()
highest_precision_model = results_df['Precision'].idxmax()
highest_recall_model = results_df['Recall'].idxmax()
highest_f1_model = results_df['F1 Score'].idxmax()
highest_auc_model = results_df['AUC Score'].idxmax()

# Display the highest metrics models
print("\nModels with highest metrics:")
print(f"Highest Accuracy: {highest_accuracy_model} with {results_df['Accuracy'][highest_accuracy_model]}")
print(f"Highest Precision: {highest_precision_model} with {results_df['Precision'][highest_precision_model]}")
print(f"Highest Recall: {highest_recall_model} with {results_df['Recall'][highest_recall_model]}")
print(f"Highest F1 Score: {highest_f1_model} with {results_df['F1 Score'][highest_f1_model]}")
print(f"Highest AUC Score: {highest_auc_model} with {results_df['AUC Score'][highest_auc_model]}")
