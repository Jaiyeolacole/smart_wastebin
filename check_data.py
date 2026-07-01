import os
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

for cls in ["plastic", "non_plastic"]:
    folder = f"dataset/{cls}"
    all_files = os.listdir(folder)
    image_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]
    non_image_files = [f for f in all_files if not f.lower().endswith(IMAGE_EXTENSIONS)]

    bad_files = []
    for fname in image_files:
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            bad_files.append(fname)

    print(f"\n{cls}: {len(all_files)} total files, {len(image_files)} images, {len(non_image_files)} non-image files (e.g. .xml), {len(bad_files)} genuinely corrupt images")

    for f in non_image_files:
        os.remove(os.path.join(folder, f))

    for f in bad_files:
        os.remove(os.path.join(folder, f))

print("\nCleanup done.")