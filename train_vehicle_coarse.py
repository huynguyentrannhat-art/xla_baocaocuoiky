"""Train YOLO classification model for supplied 4-class vehicle crop dataset.
This is a coarse classifier, NOT the project's 48-class object detector.
"""
import argparse, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base", default="yolo26n-cls.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    a=p.parse_args()
    data=ROOT/"dataset"/"vehicle_classification"
    required=[data/"train",data/"val",data/"test"]
    if not all(x.is_dir() for x in required):
        raise SystemExit("Dataset vehicle_classification thiếu train/val/test")
    from ultralytics import YOLO
    model=YOLO(a.base)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, device=a.device, workers=a.workers,
                project=str(ROOT/"runs"), name="vehicle_coarse_cls",
                patience=20, plots=True)
    best=ROOT/"runs"/"vehicle_coarse_cls"/"weights"/"best.pt"
    if not best.exists(): raise SystemExit("Không tìm thấy best.pt")
    target=ROOT/"weights"/"vehicles_vn_coarse_cls.pt"
    target.parent.mkdir(exist_ok=True,parents=True); shutil.copy2(best,target)
    print("Saved:",target)
if __name__=="__main__": main()
