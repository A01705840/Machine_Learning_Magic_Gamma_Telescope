import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
epoc_limit = 1000

col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

df = pd.read_csv('magic04.data', header=None, names=col_names)

# Mezcla los datos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_to_norm = df.drop('class', axis=1)
class_col = df['class']

df_norm = normalize(df_to_norm)

df_norm['class'] = class_col
df = df_norm
df['class'] = df['class'].map({'g': 0, 'h': 1})

df_train = df[len(df)//2:]
df_test = df[:len(df)//2]
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

theta = np.random.rand(df_train.shape[1]-1)
b = np.random.rand(1)[0]
print("Initial coefficients:", theta, b)

error = 1
epoc = 0

x_train = df_train.drop('class', axis=1).values
y_train = df_train['class'].values
while (error > 0.01 and epoc < epoc_limit):
  theta, b = update(x_train, theta, b, y_train)
  error = binary_cross_entropy(x_train, theta, b, y_train)
  if epoc % 100 == 0:  
        print(f"Epoc: {epoc}, Error: {error:.6f}")
    
  epoc += 1
	
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
