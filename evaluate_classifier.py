from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ultralytics import YOLO
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# 1. Duong dan du lieu va trong so
DATA_DIR = Path("dataset/vehicle_classification/test")
WEIGHTS = Path("runs/vehicle_coarse_cls/weights/best.pt")

# 2. Kiem tra duong dan
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Khong tim thay dataset: {DATA_DIR}")

if not WEIGHTS.is_file():
    raise FileNotFoundError(f"Khong tim thay weights: {WEIGHTS}")

# 3. Nap mo hinh
model = YOLO(str(WEIGHTS))

class_names = ["bus", "car", "truck", "van"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

y_true = []
y_pred = []

# 4. Duyet tung lop trong tap Test
for class_name in class_names:
    folder = DATA_DIR / class_name

    if not folder.is_dir():
        print("Thieu thu muc:", folder)
        continue

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    print(f"Dang danh gia {class_name}: {len(images)} anh")

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=224,
            verbose=False
        )[0]

        if result.probs is None:
            continue

        predicted_id = int(result.probs.top1)
        predicted_name = str(
            model.names[predicted_id]
        ).lower()

        if predicted_name not in class_names:
            continue

        y_true.append(class_name)
        y_pred.append(predicted_name)

# 5. Kiem tra ket qua
if not y_true:
    raise RuntimeError("Khong co anh nao duoc danh gia!")

# 6. Classification Report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_true,
    y_pred,
    labels=class_names,
    zero_division=0
))

accuracy = accuracy_score(y_true, y_pred)
print("TEST ACCURACY:", round(accuracy * 100, 2), "%")
print("Tong so anh:", len(y_true))

# 7. Confusion Matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=class_names
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Nhãn dự đoán")
plt.ylabel("Nhãn thực tế")
plt.title("Confusion Matrix - Vehicle Classification Test")
plt.tight_layout()

plt.savefig(
    "confusion_matrix_vehicle_test.png",
    dpi=200
)

plt.show()

print("\nDa luu: confusion_matrix_vehicle_test.png")
