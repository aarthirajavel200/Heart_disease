import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load the dataset
df = pd.read_csv("D://heart.csv")
df.columns = [
    'Age (years)',
    'Sex (0 = Female, 1 = Male)',
    'Chest Pain Type',
    'Resting Blood Pressure (mm Hg)',
    'Serum Cholesterol (mg/dl)',
    'Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)',
    'Resting ECG Results',
    'Maximum Heart Rate Achieved',
    'Exercise-Induced Angina (1 = Yes, 0 = No)',
    'ST Depression (Oldpeak)',
    'Slope of Peak Exercise ST Segment',
    'Major Vessels Colored by Fluoroscopy',
    'Thalassemia',
    'Heart Disease (1 = Yes, 0 = No)'
]

X = df.drop('Heart Disease (1 = Yes, 0 = No)', axis=1)
y = df['Heart Disease (1 = Yes, 0 = No)']


# Store feature names for reference
feature_names = list(X.columns)
print("Features used for training:", feature_names)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define models with class_weight for imbalance
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "SVM": SVC(probability=True, class_weight='balanced')
}

# Step 6: Train, Evaluate, and Compare Models
accuracies = {}
reports = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease Present"])
    
    accuracies[name] = acc
    reports[name] = report

# Step 7: Choose Best Model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

print("\n✅ Model Accuracies:")
for name, acc in accuracies.items():
    print(f"{name}: {acc:.4f}")

print(f"\n🏆 Best Model: {best_model_name}")
print("\n📊 Classification Report for Best Model:")
print(reports[best_model_name])

# Step 8: Save the best model, scaler, and feature names
pickle.dump(best_model, open("best_heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(feature_names, open("feature_names.pkl", "wb"))

print("\n✅ Model, Scaler, and Feature Names saved successfully.")
