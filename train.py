import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data_dir = r'C:\Users\gaado\OneDrive\المستندات\مجلد جديد\data\train'

data_file = "data.txt"
data = np.loadtxt(data_file)

X = data[:, :-1]
y = data[:, -1]

emotions = sorted([
    folder for folder in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, folder))
    and folder.lower() in ['angry', 'happy', 'sad']
])

print("Classes used for training:", emotions)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "emotions": emotions
    }, f)

print("Model saved successfully as model.pkl")