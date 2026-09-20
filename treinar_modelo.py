import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "dados_gestos"
MODEL_OUTPUT_PATH = "modelo_gestos.p"

# 1. Varredura e carregamento dos arquivos .npy
labels_dict = {}
sequences, labels = [], []

# Identifica as classes pelas pastas existentes
actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

for idx, action in enumerate(actions):
    labels_dict[idx] = action
    action_path = os.path.join(DATA_PATH, action)
    npy_files = [f for f in os.listdir(action_path) if f.endswith(".npy")]
    
    for file_name in npy_files:
        res = np.load(os.path.join(action_path, file_name))
        # Achata a matriz (30, 63) para um vetor de 1890 atributos
        sequences.append(res.flatten())
        labels.append(idx)

X = np.array(sequences)
y = np.array(labels)

print(f"Total de amostras carregadas: {X.shape[0]}")
print(f"Dimensões do vetor de entrada: {X.shape[1]} (30 frames x 63 pontos)")
print(f"Classes identificadas: {labels_dict}")

# 2. Separação em Treino e Teste (80% treino / 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Instanciação e Treinamento do Random Forest
print("\nTreinando o Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Avaliação de Desempenho
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%\n")
print("Relatório de Classificação detalhado:")
print(classification_report(y_test, y_pred, target_names=actions))

# 5. Exportação do modelo treinado junto com o mapeamento de classes
payload = {
    "model": clf,
    "labels_dict": labels_dict
}

with open(MODEL_OUTPUT_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"Modelo treinado e salvo com sucesso em '{MODEL_OUTPUT_PATH}'!")