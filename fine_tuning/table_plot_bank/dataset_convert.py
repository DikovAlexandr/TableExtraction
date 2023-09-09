from PIL import Image
import os

input_folder = os.path.join(os.getcwd(), "train")
image_files = os.listdir(input_folder)

for i, image_file in enumerate(image_files):
    input_image_path = os.path.join(input_folder, image_file)
    img = Image.open(input_image_path)
    img = img.convert("L")
    new_filename = f"{i + 1:06d}.jpg"
    img.save(os.path.join(input_folder, new_filename))
    os.remove(input_image_path)

print("Conver ready")