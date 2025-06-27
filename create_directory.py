import os


def create_required_directories():
    """Create all required directories for the application"""

    directories = [
        'static',
        'static/uploads',
        'static/images',
        'datasets',
        'datasets/train',
        'datasets/train/images',
        'datasets/valid',
        'datasets/valid/images',
        'datasets/test',
        'datasets/test/images',
        'runs',
        'runs/detect',
        'runs/detect/Accident',
        'runs/detect/Accident/weights',
        'runs/detect/crack',
        'runs/detect/crack/weights'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


if __name__ == "__main__":
    print("Creating required directories for Accident Severity Detection application...")
    create_required_directories()
    print("Directory setup complete!")
