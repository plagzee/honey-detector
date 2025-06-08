from PIL import Image
import os

# Root folder of dataset
root_dir = "dataset"

# Loop through both classes: honey and not_honey
for category in ['honey', 'not_honey']:
    folder = os.path.join(root_dir, category)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            # Try to open and verify image
            with Image.open(path) as img:
                img.verify()  # Checks if image can be opened
        except Exception as e:
            print(f"❌ Removing corrupted file: {path} ({e})")
            os.remove(path)
