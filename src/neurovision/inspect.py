from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/raw")

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

for class_name in CLASSES:
    class_dir = DATA_DIR / class_name
    
    if not class_dir.exists():
        print(f"Missing folder: {class_dir}")
        continue
    
    images = [
        file
        for file in class_dir .iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]
    
    print(f"{class_name}: {len(images)} images")
    
    # for image in images[:3]:
        # print(f" - {image.name}")