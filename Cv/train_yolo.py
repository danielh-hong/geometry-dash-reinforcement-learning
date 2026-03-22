from ultralytics import YOLO

def main():
    # Load a stronger model (yolov8s is a great balance of accuracy vs speed)
    model = YOLO("yolo11n.pt")  
    
    # Use the model
    results = model.train(
        data="data/data.yaml",   # path to dataset YAML
        epochs=300,             # number of training epochs
        patience=50,             # early stopping if no improvement for 50 epochs
        batch=-1,                # AutoBatch: scales to your GPU memory automatically
        imgsz=640,               # training image size
        device="cuda",           # device to run on
        workers=0,               # avoid Windows multiprocessing WinError 1455
        optimizer="AdamW",       # Better optimizer for custom datasets
        cos_lr=True,             # Cosine learning rate scheduling
        close_mosaic=10          # Disable mosaic in final 10 epochs for better fine-tuning
    )

if __name__ == "__main__":
    main()
