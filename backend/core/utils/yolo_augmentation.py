import os
import cv2
import numpy as np
import albumentations as A
import shutil
import os
from django.conf import settings

# Transformaciones con soporte YOLO
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Cargar bounding boxes en formato YOLO
def load_yolo_labels(label_path):
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, x, y, w, h = map(float, parts)
                    bboxes.append([int(cls_id), x, y, w, h])
    return bboxes

# Guardar imagen y etiqueta .txt
def save_augmented_data(image, bboxes, output_img_path, output_lbl_path):
    cv2.imwrite(output_img_path, image)
    with open(output_lbl_path, 'w') as f:
        for bbox in bboxes:
            cls_id, x, y, w, h = bbox
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# Generador de datos sintéticos
def generate_augmented_yolo_data(
    input_img_dir,
    input_lbl_dir,
    output_img_dir,
    output_lbl_dir,
    image_size=(416, 416),
    count=500,
    dataset_id=None
):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    img_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.png'))]
    counter = 0

    while counter < count:
        for img_file in img_files:
            img_path = os.path.join(input_img_dir, img_file)
            lbl_path = os.path.join(input_lbl_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, image_size)
            bboxes = load_yolo_labels(lbl_path)
            if not bboxes:
                continue

            class_labels = [bbox[0] for bbox in bboxes]
            yolo_boxes = [bbox[1:] for bbox in bboxes]

            try:
                augmented = augment(image=image, bboxes=yolo_boxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_classes = augmented['class_labels']
            except Exception as e:
                print(f"❌ Error con {img_file}: {e}")
                continue

            # Salida
            out_img_path = os.path.join(output_img_dir, f"synthetic_{counter}.jpg")
            out_lbl_path = os.path.join(output_lbl_dir, f"synthetic_{counter}.txt")

            full_bboxes = [[cls] + list(bbox) for cls, bbox in zip(aug_classes, aug_bboxes)]
            save_augmented_data(aug_img, full_bboxes, out_img_path, out_lbl_path)
            counter += 1

            if counter >= count:
                break
    preparar_estructura_yolo(os.path.join(settings.MEDIA_ROOT, f'dataset_yolo_{dataset_id}'))

    print(f"✅ {counter} imágenes sintéticas generadas en {output_img_dir}")



def preparar_estructura_yolo(base_dir):
    images_src = os.path.join(base_dir, "images")
    labels_src = os.path.join(base_dir, "labels")

    # Crear estructura de destino
    train_img = os.path.join(images_src, "train")
    val_img = os.path.join(images_src, "val")
    train_lbl = os.path.join(labels_src, "train")
    val_lbl = os.path.join(labels_src, "val")

    os.makedirs(train_img, exist_ok=True)
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(train_lbl, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    # Mover todos los archivos existentes a train/
    for f in os.listdir(images_src):
        if f.endswith((".jpg", ".png")):
            shutil.move(os.path.join(images_src, f), os.path.join(train_img, f))
    for f in os.listdir(labels_src):
        if f.endswith(".txt"):
            shutil.move(os.path.join(labels_src, f), os.path.join(train_lbl, f))

    # Copiar train -> val si quieres evitar errores de validación
    for f in os.listdir(train_img):
        shutil.copy(os.path.join(train_img, f), os.path.join(val_img, f))
    for f in os.listdir(train_lbl):
        shutil.copy(os.path.join(train_lbl, f), os.path.join(val_lbl, f))
