from celery import shared_task
from .models import ModeloTrain, ModeloDataset, GeneratedDataset
from core.utils.train_model import train_cnn_or_lstm, train_naive_bayes_classifier
from .utils.deep_smote_generator import generate_synthetic_tensor_dataset
from .utils.deep_smote_audio import generate_synthetic_audio_dataset
from django.contrib.auth.models import User
import time
from .utils.kmeans import train_kmeans_model, convert_numpy_types
from core.utils.train_model import train_svm_classifier, load_svm_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import joblib

from celery import shared_task
from .models import ModeloTrain, ModeloDataset, GeneratedDataset
from core.utils.train_model import train_cnn_or_lstm, train_naive_bayes_classifier
from .utils.deep_smote_generator import generate_synthetic_tensor_dataset
from .utils.deep_smote_audio import generate_synthetic_audio_dataset
from django.contrib.auth.models import User
from .utils.kmeans import train_kmeans_model, convert_numpy_types
from core.utils.train_model import train_svm_classifier, load_svm_data
from core.utils.train_model import train_yolo_model  
from django.conf import settings
import time, os, joblib, json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

@shared_task
def entrenar_modelo_task(modelo_train_id, params=None):
    print("📢 Celery task iniciada")
    try:
        modelo_train = ModeloTrain.objects.get(id=modelo_train_id)
        print(f"🧠 Entrenamiento solicitado: {modelo_train.training_name}")
        modelo_train.status = 'en progreso'
        modelo_train.save()

        modelo = modelo_train.model
        dataset = modelo.dataset
        model_type = modelo.model_type.lower()

        # Carpeta correcta para cada tipo
        # Determinar dataset_folder según el tipo de modelo
        if model_type == "yolo":
            dataset_folder = os.path.join(settings.BASE_DIR, dataset.folder_paths.get("video_augmented", ""))
        elif model_type in ["naive_bayes", "kmeans", "svm"]:
            dataset_folder = dataset.folder_paths.get("cnn")
        else:
            dataset_folder = dataset.folder_paths.get(model_type)

        # Verificación
        if not dataset_folder or not os.path.exists(dataset_folder):
            raise ValueError(f"No se encontró folder_path para el modelo: {model_type}")


        if model_type == "naive_bayes":
            var_smoothing = modelo_train.var_smoothing or 1e-9
            test_split = modelo_train.test_split or 0.2
            model_path, accuracy, combined_data = train_naive_bayes_classifier(
                experiment_name=modelo_train.training_name,
                dataset_folder=dataset_folder,
                var_smoothing=var_smoothing,
                test_split=test_split
            )
            loss = None

        elif model_type == "kmeans":
            model_path, combined_data = train_kmeans_model(
                experiment_name=modelo_train.training_name,
                dataset_folder=dataset_folder,
                n_clusters=modelo_train.n_clusters or 3
            )
            accuracy = None
            loss = None

        elif model_type == "svm":
            test_split = params.get("test_split", 0.2)
            kernel = params.get("kernel", "rbf")
            C = params.get("C", 1.0)
            X, y, class_names = load_svm_data(dataset_folder)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
            model = SVC(kernel=kernel, C=C, probability=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            model_dir = "media/models_trained"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{modelo_train.id}_svm_model.pkl")
            joblib.dump(model, model_path)
            loss = None
            combined_data = {
                "confusion_matrix": cm.tolist(),
                "class_names": class_names
            }

        elif model_type == "yolo":
            video_folder = dataset.folder_paths.get("video_augmented") 
            dataset_folder = os.path.dirname(video_folder)             
            class_names = ['casco', 'mascarilla', 'chaleco']  
            imgsz = params.get("imgsz", 640)
            lr0 = params.get("lr0", 0.01)
            model_size = params.get("model_size", "yolov8n.pt")

            model_path = train_yolo_model(
                dataset_folder=dataset_folder,
                class_names=class_names,
                experiment_name=modelo_train.training_name,
                model_size=model_size,
                epochs=modelo_train.epochs or 50,
                imgsz=imgsz,
                lr0=lr0
            )

            accuracy = None
            loss = None
            combined_data = {}

        else:
            model_path, accuracy, loss, combined_data = train_cnn_or_lstm(
                experiment_name=modelo_train.training_name,
                data_type=model_type,
                dataset_folder=dataset_folder,
                epochs=modelo_train.epochs,
                batch_size=modelo_train.batch_size
            )

        modelo_train.model_path = model_path
        modelo_train.accuracy = accuracy
        modelo_train.loss = loss
        modelo_train.combined_data = convert_numpy_types(combined_data)
        modelo_train.status = 'completado'
        modelo_train.save()
        print("✅ Modelo guardado correctamente")

    except Exception as e:
        print("❌ Error durante el entrenamiento:", str(e))
        modelo_train.status = 'fallo'
        modelo_train.save()
        raise e


@shared_task
def generar_dataset_task(dataset_id):
    from core.utils.yolo_augmentation import generate_augmented_yolo_data
    from django.conf import settings
    import os

    try:
        dataset = GeneratedDataset.objects.get(id=dataset_id)
        print(f"📦 Generando dataset: {dataset.name}")
        dataset.status = 'en_progreso'
        dataset.save()

        if dataset.data_type == "sensor":
            output_paths = generate_synthetic_tensor_dataset(name=dataset.name, sample_count=dataset.sample_count)

        elif dataset.data_type == "audio":
            output_paths = generate_synthetic_audio_dataset(name=dataset.name, sample_count=dataset.sample_count)

        elif dataset.data_type == "video":
            # 🔁 Tratar YOLO como tipo "video"
            input_img_dir = os.path.join(settings.MEDIA_ROOT, 'imagenes/images/train')
            input_lbl_dir = os.path.join(settings.MEDIA_ROOT, 'imagenes/labels/train')
            output_img_dir = os.path.join(settings.MEDIA_ROOT, f'dataset_yolo_{dataset_id}/images')
            output_lbl_dir = os.path.join(settings.MEDIA_ROOT, f'dataset_yolo_{dataset_id}/labels')

            count = generate_augmented_yolo_data(
                input_img_dir=input_img_dir,
                input_lbl_dir=input_lbl_dir,
                output_img_dir=output_img_dir,
                output_lbl_dir=output_lbl_dir,
                count=dataset.sample_count,
                dataset_id=dataset_id  # 👈 aquí lo pasas

            )

            output_paths = {
                "video_augmented": os.path.relpath(output_img_dir, settings.BASE_DIR),
                "yolo_labels": os.path.relpath(output_lbl_dir, settings.BASE_DIR)
            }

        else:
            raise ValueError("Tipo de dato no soportado")

        dataset.folder_paths = {
            "cnn": output_paths.get("cnn", [None])[0],
            "lstm": output_paths.get("lstm", [None])[0],
            "video_augmented": output_paths.get("video_augmented"),
            "yolo_labels": output_paths.get("yolo_labels")
        }

        dataset.status = 'completado'
        dataset.save()
        print("✅ Dataset generado correctamente")

    except Exception as e:
        print("❌ Error al generar dataset:", str(e))
        if 'dataset' in locals():
            dataset.status = 'fallido'
            dataset.save()
        raise e





