# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# 2. Load Dataset
def load_data(path):
    df = pd.read_csv(path)
    return df

# 3. Preprocess Data
def preprocess_data(df):
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'])
    df = pd.get_dummies(df, columns=['Department', 'EducationField', 'MaritalStatus'], drop_first=False)
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 4. Train & Evaluate Multiple Models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=2),
        'SVC': SVC(probability=True),
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(),
        'AdaBoost': AdaBoostClassifier(n_estimators=50)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Accuracy': acc})
        print(f"\n{name} Accuracy: {acc:.4f}")

    return pd.DataFrame(results).sort_values(by='Accuracy', ascending=False), models

# 5. Plot Accuracy Comparison
def plot_accuracy(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.show()

# 6. Final Evaluation of Best Model
def evaluate_best_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nBest Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# 7. Save the Best Model
def save_model(model, filename="employee_attrition_model.joblib"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# 8. Main Workflow
def main():
    df = load_data('IBM.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    results_df, models = evaluate_models(X_train, X_test, y_train, y_test)
    plot_accuracy(results_df)

    # Select best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    evaluate_best_model(best_model, X_test, y_test)
    save_model(best_model)

if __name__ == "__main__":
    main()
