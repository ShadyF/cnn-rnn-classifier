from PIL import Image
import os
import shutil

dataset_train_path = './tiny-imagenet-100-A/train/'
folders = [folder for folder in os.listdir(dataset_train_path)]

for folder in folders:
    label_dir = os.path.join(dataset_train_path, folder)
    images_dir = os.path.join(label_dir, 'images/')
    images = [image for image in os.listdir(images_dir)]

    for img in images:
        img_path = os.path.join(images_dir, img)
        relocated_img = Image.open(img_path)
        relocated_img.save(os.path.join(label_dir, img))

    shutil.rmtree(images_dir)
    os.unlink(os.path.join(label_dir, folder + '_boxes.txt'))

dataset_val_path = './tiny-imagenet-100-A/val/'
val_images_dir = os.path.join(dataset_val_path, 'images')

labels_dict = {}
with open(os.path.join(dataset_val_path, 'val_annotations.txt'), 'r') as f:
    for line in f:
        line_list = line.strip().split()
        label = line_list[1]
        labels_dict[line_list[0]] = label
        try:
            os.mkdir(os.path.join(dataset_val_path, label))
        except OSError:
            pass

val_images = [img for img in os.listdir(val_images_dir)]

for val_image in val_images:
    img_path = os.path.join(val_images_dir, val_image)
    label = labels_dict[val_image]

    relocated_img = Image.open(img_path)
    relocated_img.save(os.path.join(dataset_val_path, label, val_image))

shutil.rmtree(val_images_dir)
os.unlink(os.path.join(dataset_val_path, 'val_annotations.txt'))
