
from pathlib import Path
import csv
from collections import Counter
from ultralytics import YOLO

DATA_DIR = Path("dataset/vehicle_classification/test")
WEIGHTS = Path("runs/vehicle_coarse_cls/weights/best.pt")

model = YOLO(str(WEIGHTS))
classes = ["bus", "car", "truck", "van"]
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = []

for true_class in classes:
    folder = DATA_DIR / true_class

    for image_path in sorted(folder.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in extensions:
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=224,
            verbose=False
        )[0]

        if result.probs is None:
            continue

        pred_id = int(result.probs.top1)
        pred_class = str(model.names[pred_id]).lower()
        confidence = float(result.probs.top1conf.item())

        records.append({
            "image": str(image_path),
            "actual": true_class,
            "predicted": pred_class,
            "confidence": round(confidence, 4),
            "correct": pred_class == true_class
        })

# Luu toan bo ket qua
with open("test_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image", "actual", "predicted", "confidence", "correct"
    ])
    writer.writeheader()
    writer.writerows(records)

# Thong ke
correct_count = sum(r["correct"] for r in records)
print("TONG SO ANH:", len(records))
print("DU DOAN DUNG:", correct_count)
print("ACCURACY:", round(correct_count / len(records) * 100, 2), "%")

print("\n===== SO LUONG DU DOAN SAI THEO LOP =====")
for c in classes:
    wrong = [
        r for r in records
        if r["actual"] == c and not r["correct"]
    ]
    print(f"{c}: {len(wrong)} anh sai")

print("\n===== 20 ANH DU DOAN SAI DAU TIEN =====")
for r in [x for x in records if not x["correct"]][:20]:
    print(
        f"Thuc te: {r['actual']:5} | "
        f"Du doan: {r['predicted']:5} | "
        f"Confidence: {r['confidence']:.2f} | "
        f"Anh: {Path(r['image']).name}"
    )

print("\nDa luu ket qua vao test_predictions.csv")
