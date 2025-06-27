import os
from ultralytics import YOLO
import argparse
from datetime import datetime


def train_model(data_yaml=os.path.join('datasets', 'data.yaml'), epochs=100, batch_size=16, img_size=640,
                weights='yolov8n.pt'):
    """Train a YOLOv8 model on the accident severity dataset"""
    # Create timestamp for run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"accident_severity_{timestamp}"

    # Ensure data_yaml path is correct
    if not os.path.exists(data_yaml):
        print(f"Warning: Data file {data_yaml} does not exist. Checking alternative paths...")
        alt_path = os.path.join(os.getcwd(), 'datasets', 'data.yaml')
        if os.path.exists(alt_path):
            data_yaml = alt_path
            print(f"Found data file at: {data_yaml}")
        else:
            print(f"Error: Could not find data.yaml file. Please check the path.")
            return None

    # Load the model
    model = YOLO(weights)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name=run_name,
        patience=20,  # Early stopping patience
        save=True,  # Save best model
        device='0' if torch.cuda.is_available() else 'cpu'
    )

    # Validate the model
    val_results = model.val()

    return {
        'model': model,
        'training_results': results,
        'validation_results': val_results,
        'run_name': run_name
    }


def main():
    parser = argparse.ArgumentParser(description='Train Accident Severity Detection Model')
    parser.add_argument('--data', type=str, default=os.path.join('datasets', 'data.yaml'),
                        help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Initial weights')
    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} does not exist")
        return

    # Print training configuration
    print("=== Training Configuration ===")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img_size}")
    print(f"Initial weights: {args.weights}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load dataset info
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"Dataset: {data_config.get('names', ['Unknown'])}")
    print(f"Train images: {data_config.get('train', 'Unknown')}")
    print(f"Val images: {data_config.get('val', 'Unknown')}")
    print(f"Test images: {data_config.get('test', 'Unknown')}")
    print("============================")

    # Start training
    print("Starting training...")
    result = train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        weights=args.weights
    )

    print(f"Training completed. Results saved in runs/detect/{result['run_name']}")

    # Print validation metrics
    metrics = result['validation_results'].box
    print("\n=== Validation Metrics ===")
    print(f"mAP50: {metrics.map50:.4f}")
    print(f"mAP50-95: {metrics.map:.4f}")
    print(f"Precision: {metrics.p:.4f}")
    print(f"Recall: {metrics.r:.4f}")
    print("=========================")


if __name__ == "__main__":
    import torch
    import yaml

    main()
