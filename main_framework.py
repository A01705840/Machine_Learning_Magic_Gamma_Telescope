import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy.optimize import minimize
from sklearn import tree
import matplotlib.pyplot as plt

col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df = pd.read_csv('magic04.data', header=None, names=col_names)

# Mezcla y normaliza los datos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def log_likelihood(y_true, y_pred_prob):
    eps = 1e-10
    return np.sum(y_true * np.log(y_pred_prob + eps) + (1 - y_true) * np.log(1 - y_pred_prob + eps))

df_to_norm = df.drop('class', axis=1)
df_norm = normalize(df_to_norm)
df_norm['class'] = df['class']
df = df_norm
df['class'] = df['class'].map({'g': 0, 'h': 1})

# Divide los datos en características (X) y etiquetas (y)
X = df.drop('class', axis=1)
y = df['class']

# Divide en conjuntos de entrenamiento (60%) y prueba (40%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# El hiperparámetro n_estimators define el número de árboles en el bosque
model = RandomForestClassifier(n_estimators=250, random_state=42)

model.fit(X_train, y_train)

# Hace predicciones en los conjuntos de entrenamiento, validación y prueba
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Calcula y muestra la precisión de cada conjunto
train_accuracy = accuracy_score(y_train, y_pred_train)
val_accuracy = accuracy_score(y_val, y_pred_val)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


