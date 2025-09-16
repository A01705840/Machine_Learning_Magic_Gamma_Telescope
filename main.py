import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from scipy.optimize import minimize

def normalize(df):
  result = df.copy()
  for feature_name in df.columns:
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    # Aplica la fórmula de normalización (x - min) / (max - min)
    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
  return result

def predict(x, theta, b):
    z = np.dot(x, theta) + b
    sigmoid = np.exp(-(z))
    return 1.0 / (1.0 + np.exp(-z))

#BATCH GRADIENT DESCENT
def gradient_descent(df, y_true, theta, b, col):
  new_param = list(theta)
  for j in range(len(theta)-1):
    new_theta = 0
    for i in range(len(df)):
        y_pred = predict(df[i], theta, b)
        error = (y_pred - y_true[i])
        new_theta += error * df[i][j]
    new_param[j] = theta[j] - (learn_rate/(len(theta))) * new_theta
  return new_param

def gradient_descent_b(df, y_true, theta, b):
  new_b = b
  for i in range(len(df)):
      y_pred = predict(df[i], theta, b)
      error = (y_pred - y_true[i])
      new_b += error
  new_b = b - learn_rate/(len(theta)) * new_b
  return new_b

#Aplica GD para actualizar los parámetros
def update(df, theta, b, y_true):
  theta_new = gradient_descent(df, y_true, theta, b, 0)
  b_new = gradient_descent_b(df, y_true, theta, b)
  return theta_new, b_new


def binary_cross_entropy(df, theta, b, y_true):
    cost = 0
    m = len(y_true)
    eps = 1e-10
    for i in range(m):
        y_pred = predict(df[i], theta, b)
        cost += -y_true[i]*np.log(y_pred+eps) - (1-y_true[i])*np.log(1-y_pred+eps)
    return cost/m

def accuracy(y_pred, y_true):
    # Convierte las probabilidades a predicciones binarias (0 o 1)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    # Calcula la media de las coincidencias
    accuracy_score = np.mean(y_pred_binary == y_true)

    return accuracy_score

learn_rate = 0.005
print('learning rate:', learn_rate)
epoc_limit = 6000

col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

# Carga los datos
df = pd.read_csv('magic04.data', header=None, names=col_names)

# Mezcla los datos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_to_norm = df.drop('class', axis=1)
class_col = df['class']

df_norm = normalize(df_to_norm)

df_norm['class'] = class_col
df = df_norm
df['class'] = df['class'].map({'g': 0, 'h': 1})


# Divide los datos en un 80% para entrenamiento+validación y un 20% para prueba
train_val_split = int(0.8 * len(df))
df_train_val = df[:train_val_split]
df_test = df[train_val_split:]

# Ahora divide el conjunto de entrenamiento+validación en un 75% para entrenamiento y 25% para validación
val_split = int(0.75 * len(df_train_val))
df_train = df_train_val[:val_split]
df_val = df_train_val[val_split:]

print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)

# Actualiza las variables de datos para el entrenamiento y crea las de validación
x_train = df_train.drop('class', axis=1).values
y_train = df_train['class'].values

x_val = df_val.drop('class', axis=1).values
y_val = df_val['class'].values

# Inicializa las listas para guardar los errores y la precisión por época
train_errors = []
val_errors = []
train_accuracies = []
val_accuracies = []

theta = np.random.rand(x_train.shape[1])
b = np.random.rand(1)[0]
print("Initial coefficients:", theta, b)

error = 1
epoc = 0
while (error > 0.01 and epoc < epoc_limit):
    # Actualizar los parámetros
    theta, b = update(x_train, theta, b, y_train)

    # Calcular y guardar el error de entrenamiento
    train_error = binary_cross_entropy(x_train, theta, b, y_train)
    train_errors.append(train_error)

    # Calcular y guardar el error de validación
    val_error = binary_cross_entropy(x_val, theta, b, y_val)
    val_errors.append(val_error)

    # Calcular y guardar la precisión en ambos conjuntos
    y_pred_train = predict(x_train, theta, b)
    train_accuracy = accuracy(y_pred_train, y_train)
    train_accuracies.append(train_accuracy)

    y_pred_val = predict(x_val, theta, b)
    val_accuracy = accuracy(y_pred_val, y_val)
    val_accuracies.append(val_accuracy)
    
    # Imprime el progreso cada 100 épocas
    if epoc % 100 == 0: 
      print(f"Epoc: {epoc}, Train Error: {train_error:.6f}, Val Error: {val_error:.6f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    
    error = train_error
    epoc += 1
    

# --- Código para graficar las curvas de error ---
plt.figure(figsize=(10, 6))

# Grafica el error de entrenamiento y de validación
plt.plot(train_errors, label='Error de Entrenamiento')
plt.plot(val_errors, label='Error de Validación')

# Agrega títulos y etiquetas
plt.title('Curvas de Aprendizaje: Error vs. Épocas')
plt.xlabel('Épocas')
plt.ylabel('Error (Binary Cross-Entropy)')
plt.legend()
plt.grid(True)

# Muestra el gráfico
plt.show()


plt.figure(figsize=(10, 6))

# Grafica el acc de entrenamiento y de validación
plt.plot(train_accuracies, label='Error de Entrenamiento')
plt.plot(val_accuracies, label='Error de Validación')

# Agrega títulos y etiquetas
plt.title('Curvas de Aprendizaje: Error vs. Épocas')
plt.xlabel('Épocas')
plt.ylabel('Error (Binary Cross-Entropy)')
plt.legend()
plt.grid(True)

# Muestra el gráfico
plt.show()

print("Final coefficients:", theta, b)

x_test = df_test.drop('class', axis=1).values
y_test = df_test['class'].values

y_pred_test = predict(x_test, theta, b)
test_accuracy = accuracy(y_pred_test, y_test)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Gráfico para la Clase 0 Verdadera (gamma-ray)
ax1.hist(y_pred_test[y_test == 0], bins=50, alpha=0.9, color='blue')
ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
ax1.set_title('Distribution of True Class 0 (g)')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Number of Samples')
ax1.grid(True)
ax1.legend()

# Gráfico para la Clase 1 Verdadera (hadron)
ax2.hist(y_pred_test[y_test == 1], bins=50, alpha=0.9, color='red')
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
ax2.set_title('Distribution of True Class 1 (h)')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Number of Samples')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Function to calculate log-likelihood for McFadden's R^2
def log_likelihood(y_true, y_pred_prob):
    eps = 1e-10
    return np.sum(y_true * np.log(y_pred_prob + eps) + (1 - y_true) * np.log(1 - y_pred_prob + eps))

# Calculate Log-Likelihood of the full model
y_pred_test_prob = predict(x_test, theta, b)
ll_model = log_likelihood(y_test, y_pred_test_prob)

# Calculate Log-Likelihood of the null model (predicting the mean)
mean_y = np.mean(y_test)
ll_null = np.sum(y_test * np.log(mean_y) + (1 - y_test) * np.log(1 - mean_y))

# Calculate McFadden's R^2
mc_r2 = 1 - (ll_model / ll_null)
print(f"McFadden's R^2: {mc_r2:.4f}")

# --- Classification Report ---
# Convert probabilities to binary predictions
y_pred_binary = (y_pred_test_prob >= 0.5).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

# Calculate other metrics
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")



# --- Código para la Curva ROC y el AUC ---

# Obtiene las probabilidades predichas del conjunto de prueba
# Tu función 'predict' ya hace esto.
y_pred_test_prob = predict(x_test, theta, b)

# Calcula la Tasa de Falsos Positivos (FPR),
# la Tasa de Verdaderos Positivos (TPR),
# y los umbrales de decisión.
# La función roc_curve de sklearn necesita los valores reales (y_true)
# y las probabilidades predichas (y_pred_prob).
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_prob)

# Calcula el Área Bajo la Curva (AUC)
roc_auc = auc(fpr, tpr)

# Imprime el valor del AUC
print(f"\nAUC-ROC Score: {roc_auc:.4f}")

# --- Código para graficar la Curva ROC ---
plt.figure(figsize=(8, 8))

# Dibuja la curva ROC
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')

# Dibuja la línea de clasificación aleatoria (curva de referencia)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (area = 0.5)')

# Agrega los títulos y etiquetas
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)

# Muestra el gráfico
plt.show()