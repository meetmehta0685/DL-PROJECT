#!/usr/bin/env python3
"""
Quick Start Script for Pneumonia Detection Project
Run this script to verify setup and see example usage.
"""

import sys
from pathlib import Path

def check_installation():
    """Check if all required packages are installed."""
    print("="*60)
    print("Checking Installation...")
    print("="*60)
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'PIL',
        'sklearn',
        'streamlit',
        'cv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed!")
        return True

def check_project_structure():
    """Check if project structure is correct."""
    print("\n" + "="*60)
    print("Checking Project Structure...")
    print("="*60)
    
    required_dirs = [
        'data',
        'notebooks',
        'src',
    ]
    
    required_files = [
        'src/dataset.py',
        'src/model.py',
        'src/train.py',
        'src/inference.py',
        'src/gradcam.py',
        'src/utils.py',
        'app.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - MISSING")
            all_good = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} - MISSING")
            all_good = False
    
    if all_good:
        print("\n✅ Project structure is correct!")
    else:
        print("\n❌ Some files/directories are missing!")
    
    return all_good

def check_dataset():
    """Check if dataset is available."""
    print("\n" + "="*60)
    print("Checking Dataset...")
    print("="*60)
    
    data_dir = Path('data/chest_xray')
    
    if not data_dir.exists():
        print("❌ Dataset directory not found!")
        print(f"Expected location: {data_dir}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        return False
    
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    for split in splits:
        split_dir = data_dir / split
        if split_dir.exists():
            for class_name in classes:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    num_images = len(list(class_dir.glob('*.jpeg'))) + \
                                len(list(class_dir.glob('*.jpg'))) + \
                                len(list(class_dir.glob('*.png')))
                    print(f"✓ {split}/{class_name}: {num_images} images")
                else:
                    print(f"✗ {split}/{class_name} - MISSING")
        else:
            print(f"✗ {split}/ - MISSING")
    
    return True

def show_next_steps():
    """Show next steps for the user."""
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    
    print("""
1. EXPLORE DATA:
   jupyter notebook notebooks/01_data_exploration.ipynb

2. PREPROCESS DATA:
   jupyter notebook notebooks/02_preprocessing.ipynb

3. TRAIN MODEL:
   jupyter notebook notebooks/03_training.ipynb
   
   OR via script:
   python -c "from src.train import train_model; train_model()"

4. RUN WEB APP:
   streamlit run app.py

5. MAKE PREDICTIONS:
   from src.inference import PneumoniaPredictor
   predictor = PneumoniaPredictor('checkpoints/best_model.pt')
   result = predictor.predict('path/to/image.jpg')
    """)

def main():
    """Main function."""
    print("\n🫁 PNEUMONIA DETECTION PROJECT - SETUP VERIFICATION\n")
    
    # Check installation
    install_ok = check_installation()
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if install_ok and structure_ok:
        print("✅ Setup is complete!")
        if not dataset_ok:
            print("⚠️  Dataset not found - please download it to proceed with training")
        show_next_steps()
    else:
        print("❌ Setup is incomplete. Please fix the issues above.")
    
    print("\n" + "="*60)
    print("For more information, see README.md")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
