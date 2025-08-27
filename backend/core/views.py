from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status 
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework import viewsets, permissions
from .models import GeneratedDataset, ModeloDataset, ModeloTrain, PasswordResetCode
from rest_framework import serializers
from .serializers import GeneratedDatasetSerializer, ModeloDatasetSerializer, ModeloTrainSerializer
from .utils.deep_smote_generator import generate_synthetic_tensor_dataset
from .utils.deep_smote_audio import generate_synthetic_audio_dataset
from core.utils.train_model import train_cnn_or_lstm
from rest_framework.views import APIView
import os, random, numpy as np, torch, time
from core.model_definitions import get_cnn_model, get_lstm_model
import plotly.graph_objs as go
from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import send_mail
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.db.models import Max
import shutil
from .tasks import entrenar_modelo_task, generar_dataset_task
import json
from core.utils.kmeans import run_kmeans_clustering
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import base64
import io
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_protected_data(request):
    return Response({'message': 'Hola, estás autenticado correctamente 🎉'})

@api_view(['POST'])
def logout_view(request):
    try:
        refresh_token = request.data["refresh"]
        token = RefreshToken(refresh_token)
        token.blacklist()
        return Response({"detail": "Logout exitoso"}, status=status.HTTP_205_RESET_CONTENT)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)



@api_view(['POST'])
def register(request):
    username = request.data.get('email')
    email = request.data.get('email')
    first_name = request.data.get('first_name')
    last_name = request.data.get('last_name')
    password = request.data.get('password')
    confirm_password = request.data.get('confirm_password')

    if password != confirm_password:
        return Response({"error": "Las contraseñas no coinciden"}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({"error": "El nombre de usuario ya existe"}, status=400)
    
    if User.objects.filter(email=email).exists():
        return Response({"error": "El correo ya está registrado"}, status=400)

    user = User.objects.create_user(
        username=username,
        email=email,
        first_name=first_name,
        last_name=last_name,
        password=password
    )
    return Response({"message": "Usuario registrado exitosamente"})


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        data['first_name'] = self.user.first_name
        data['last_name'] = self.user.last_name
        return data

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer

@api_view(['POST'])
def send_reset_code(request):
    email = request.data.get('email')
    if not email:
        return Response({"error": "Email requerido"}, status=400)

    try:
        user = User.objects.get(email=email)
        code = f"{random.randint(100000, 999999)}"

        PasswordResetCode.objects.create(user=user, code=code)

        send_mail(
            'Código de recuperación',
            f'Tu código de recuperación es: {code}',
            'camilafuentes.1996@gmail.com',
            [email],
            fail_silently=False,
        )

        return Response({"message": "Código enviado"})
    except User.DoesNotExist:
        return Response({"error": "Usuario no encontrado"}, status=404)


@api_view(['POST'])
def reset_password(request):
    email = request.data.get('email')
    code = request.data.get('code')
    new_password = request.data.get('new_password')

    if not all([email, code, new_password]):
        return Response({"error": "Faltan campos requeridos"}, status=400)

    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        return Response({"error": "Usuario no encontrado"}, status=404)

    try:
        reset_entry = PasswordResetCode.objects.filter(
            user=user, code=code
        ).order_by('-created_at').first()

        if not reset_entry:
            return Response({"error": "Código inválido"}, status=400)

        user.set_password(new_password)
        user.save()

        PasswordResetCode.objects.filter(user=user).delete()

        return Response({"message": "Contraseña actualizada con éxito"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_recent_trainings(request):
    user = request.user
    trainings = ModeloTrain.objects.filter(user=user).order_by('-created_at')[:3]

    data = []
    for t in trainings:
        data.append({
            'nombre': t.training_name,  
            'fecha': t.created_at.strftime('%d %B, %Y'),
            'modelo': t.model.model_type,
            'accuracy': t.accuracy,  
        })

    return Response(data)

class GeneratedDatasetViewSet(viewsets.ModelViewSet):
    queryset = GeneratedDataset.objects.all()  
    serializer_class = GeneratedDatasetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return GeneratedDataset.objects.filter(user=self.request.user)

    def perform_destroy(self, instance):
        if isinstance(instance.folder_paths, dict):
            for key, path in instance.folder_paths.items():
                if path and os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)  
        instance.delete()

   

    @action(detail=True, methods=['get'], url_path='labels')
    def labels(self, request, pk=None):
        import os
        from django.conf import settings

        dataset = self.get_object()
        folder_paths = dataset.folder_paths

        cnn_folder = folder_paths.get("cnn")
        if not cnn_folder:
            return Response({"error": "No hay carpeta CNN registrada."}, status=400)

        cnn_folder_abs = os.path.join(settings.BASE_DIR, cnn_folder)
        if not os.path.exists(cnn_folder_abs):
            return Response({"error": "No existe la carpeta CNN generada."}, status=404)

        labels = [
            name for name in os.listdir(cnn_folder_abs)
            if os.path.isdir(os.path.join(cnn_folder_abs, name))
        ]

        return Response(labels)

    @action(detail=True, methods=['get'], url_path='comparison')
    def comparison(self, request, pk=None):
        dataset = self.get_object()
        folder_paths = dataset.folder_paths
        data_type = dataset.data_type

        # 📦 Comparación para VIDEO
        if data_type == "video":
            real_folder = os.path.join(settings.BASE_DIR, 'media', 'imagenes', 'images', 'train')
            gen_folder = os.path.join(settings.BASE_DIR, folder_paths.get('video_augmented', ''), 'train')
            real_imgs = [f for f in os.listdir(real_folder) if f.lower().endswith('.jpg')]
            gen_imgs = [f for f in os.listdir(gen_folder) if f.lower().endswith('.jpg')]
            if not real_imgs or not gen_imgs:
                return Response({"error": "No se encontraron imágenes reales o generadas."}, status=404)

            def encode_image(path):
                with Image.open(path) as img:
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    return base64.b64encode(buffered.getvalue()).decode()

            real_img_path = os.path.join(real_folder, random.choice(real_imgs))
            gen_img_path = os.path.join(gen_folder, random.choice(gen_imgs))

            return Response({
                "real": { "image": encode_image(real_img_path) },
                "generated": { "image": encode_image(gen_img_path) }
            })

        # 🔍 Validación para audio/sensor (necesita etiqueta)
        label = request.query_params.get('label')
        if not label:
            return Response({"error": "Label no proporcionada."}, status=400)

        # Rutas reales
        if data_type == 'audio':
            cnn_real_base = os.path.join(settings.BASE_DIR, 'media', 'tensor_cnn_audio_real')
        else:
            cnn_real_base = os.path.join(settings.BASE_DIR, 'media', 'tensor_cnn_real')
            lstm_real_base = os.path.join(settings.BASE_DIR, 'media', 'tensor_lstm_real')

        cnn_real_folder = os.path.join(cnn_real_base, label)
        cnn_gen_folder = os.path.join(settings.BASE_DIR, folder_paths.get("cnn", ""), label)

        cnn_files_real = [f for f in os.listdir(cnn_real_folder) if f.endswith(".npy")]
        cnn_files_gen = [f for f in os.listdir(cnn_gen_folder) if f.endswith(".npy")]

        if not cnn_files_real or not cnn_files_gen:
            return Response({"error": "No se encontraron espectrogramas requeridos."}, status=404)

        # CNN
        cnn_tensor_real = np.load(load_best_tensor(cnn_real_folder))
        if cnn_tensor_real.ndim == 3:
            cnn_tensor_real = cnn_tensor_real[:, :, 0]
        real_spectrogram = [go.Heatmap(z=cnn_tensor_real, colorscale='Magma')]

        cnn_tensor_gen = np.load(os.path.join(cnn_gen_folder, random.choice(cnn_files_gen)))
        if cnn_tensor_gen.ndim == 3:
            cnn_tensor_gen = cnn_tensor_gen[:, :, 0]
        gen_spectrogram = [go.Heatmap(z=cnn_tensor_gen, colorscale='Magma')]

        if data_type == 'audio':
            return Response({
                "real": {
                    "spectrogram": {"data": [d.to_plotly_json() for d in real_spectrogram]}
                },
                "generated": {
                    "spectrogram": {"data": [d.to_plotly_json() for d in gen_spectrogram]}
                }
            })

        lstm_real_folder = os.path.join(lstm_real_base, label)
        lstm_gen_path = folder_paths.get("lstm")
        if not lstm_gen_path:
            return Response({"error": "No se encontró carpeta LSTM generada."}, status=404)

        lstm_gen_folder = os.path.join(settings.BASE_DIR, lstm_gen_path, label)

        lstm_files_real = [f for f in os.listdir(lstm_real_folder) if f.endswith(".npy")]
        lstm_files_gen = [f for f in os.listdir(lstm_gen_folder) if f.endswith(".npy")]

        if not lstm_files_real or not lstm_files_gen:
            return Response({"error": "No se encontraron tensores LSTM."}, status=404)

        lstm_tensor_real = np.load(os.path.join(lstm_real_folder, random.choice(lstm_files_real)))
        y_real = lstm_tensor_real[:, 0] if lstm_tensor_real.ndim == 2 else lstm_tensor_real
        x_real = np.arange(len(y_real))
        real_temporal = [go.Scatter(x=x_real, y=y_real, mode='lines', name='Real')]

        lstm_tensor_gen = np.load(os.path.join(lstm_gen_folder, random.choice(lstm_files_gen)))
        y_gen = lstm_tensor_gen[:, 0] if lstm_tensor_gen.ndim == 2 else lstm_tensor_gen
        x_gen = np.arange(len(y_gen))
        gen_temporal = [go.Scatter(x=x_gen, y=y_gen, mode='lines', name='Generado')]

        # Final
        return Response({
            "real": {
                "temporal": {"data": [d.to_plotly_json() for d in real_temporal]},
                "spectrogram": {"data": [d.to_plotly_json() for d in real_spectrogram]}
            },
            "generated": {
                "temporal": {"data": [d.to_plotly_json() for d in gen_temporal]},
                "spectrogram": {"data": [d.to_plotly_json() for d in gen_spectrogram]}
            }
        })


    def perform_create(self, serializer):
        user = self.request.user
        name = self.request.data.get('name')
        data_type = self.request.data.get('data_type')
        sample_count = int(self.request.data.get('sample_count', 1000))
        description = self.request.data.get('description', '')

        dataset = serializer.save(
            user=user,
            name=name,
            data_type=data_type,
            sample_count=sample_count,
            description=description,
            status='pendiente'
        )
        generar_dataset_task.delay(dataset.id)


class ModeloTrainViewSet(viewsets.ModelViewSet):
    queryset = ModeloTrain.objects.all()
    serializer_class = ModeloTrainSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ModeloTrain.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        user = self.request.user
        training_name = self.request.data.get('training_name')
        modelo_id = self.request.data.get('modelo_id')
        epochs = int(self.request.data.get('epochs', 30))
        batch_size = int(self.request.data.get('batch_size', 16))

        try:
            modelo = ModeloDataset.objects.get(id=modelo_id, user=user)

            # ✅ primero definimos los parámetros según el tipo de modelo
            params = {}
            model_type = modelo.model_type.lower()

            if model_type == "naive_bayes":
                raw_params = self.request.data.get("params", {})
                if isinstance(raw_params, str):
                    raw_params = json.loads(raw_params)
                params["var_smoothing"] = float(raw_params.get("var_smoothing", 1e-9))
                params["test_split"] = float(raw_params.get("test_split", 0.2))

            elif model_type == "kmeans":
                raw_params = self.request.data.get("params", {})
                if isinstance(raw_params, str):
                    raw_params = json.loads(raw_params)
                params["n_clusters"] = int(raw_params.get("n_clusters", 3))

            elif model_type == "svm":
                raw_params = self.request.data.get("params", {})
                if isinstance(raw_params, str):
                    raw_params = json.loads(raw_params)
                params["kernel"] = raw_params.get("kernel", "rbf")
                params["C"] = float(raw_params.get("C", 1.0))
                params["test_split"] = float(raw_params.get("test_split", 0.2))


            # ✅ luego usamos los parámetros ya definidos
            modelo_train = serializer.save(
                user=user,
                training_name=training_name,
                model=modelo,
                epochs=epochs,
                batch_size=batch_size,
                status='pendiente',
                var_smoothing=params.get("var_smoothing"),
                test_split=params.get("test_split"),
                n_clusters=params.get("n_clusters"),
                kernel=params.get("kernel"),
                C=params.get("C"),

            )

            entrenar_modelo_task.delay(modelo_train.id, params)

        except ModeloDataset.DoesNotExist:
            raise serializers.ValidationError({"error": "El modelo seleccionado no existe o no pertenece al usuario."})
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise serializers.ValidationError({"error": str(e)})
    
class GetLabelsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        train_id = request.query_params.get('train_id')

        try:
            modelo_train = ModeloTrain.objects.get(id=train_id, user=request.user)

            # 1. Intentar obtener desde combined_data
            labels = (modelo_train.combined_data or {}).get("class_names")
            if labels:
                return Response({"labels": labels})

            # 2. Fallback: obtener desde el filesystem
            modelo_dataset = modelo_train.model
            generated_dataset = modelo_dataset.dataset
            model_type = modelo_dataset.model_type.lower()
            folder_path = generated_dataset.folder_paths.get(model_type)

            if not folder_path or not os.path.exists(folder_path):
                return Response({"error": "Folder path no encontrado."}, status=status.HTTP_400_BAD_REQUEST)

            labels = [
                name for name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, name))
            ]
            return Response({"labels": labels})

        except ModeloTrain.DoesNotExist:
            return Response({"error": "Entrenamiento no encontrado."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PredictRandomTensorView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_labels(self, modelo_train):
        combined = modelo_train.combined_data or {}
        labels = combined.get("class_names")
        if labels:
            print(f"✅ Labels desde combined_data: {labels}")
            return labels

        raise FileNotFoundError("❌ No se encontró archivo de clases ni en combined_data.")

    def load_random_tensor(self, folder_path, label):
        label_dir = os.path.join(folder_path, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label '{label}' no encontrado en: {label_dir}")

        files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        if not files:
            raise FileNotFoundError(f"No hay tensores para la clase '{label}'.")

        random_file = random.choice(files)
        tensor_path = os.path.join(label_dir, random_file)
        print(f"🎲 Tensor seleccionado: {random_file}")

        return np.load(tensor_path), random_file

    def validate_tensor_shape(self, tensor, model_type, data_type):
        if model_type == "lstm":
            if tensor.ndim != 2:
                raise ValueError(f"LSTM: esperado (timesteps, channels), recibido {tensor.shape}")
            if data_type == "sensor" and tensor.shape[1] != 9:
                raise ValueError(f"LSTM sensores: esperado (timesteps, 9), recibido {tensor.shape}")
            if data_type == "audio" and tensor.shape[1] not in [1, 2]:
                raise ValueError(f"LSTM audio: esperado (timesteps, 1 o 2), recibido {tensor.shape}")
        elif model_type == "cnn":
            if tensor.ndim != 3:
                raise ValueError(f"CNN: esperado (H, W, channels), recibido {tensor.shape}")
            if data_type == "sensor" and tensor.shape[2] != 9:
                raise ValueError(f"CNN sensores: esperado (64, 64, 9), recibido {tensor.shape}")
            if data_type == "audio" and tensor.shape[2] not in [1, 2]:
                raise ValueError(f"CNN audio: esperado (64, 64, 1 o 2), recibido {tensor.shape}")

    def post(self, request):
        print("🔧 [PredictRandomTensorView] POST recibido")
        train_id = request.data.get('train_id')
        label = request.data.get('label')

        if not train_id or not label:
            return Response({"error": "Falta 'train_id' o 'label'."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            modelo_train = ModeloTrain.objects.get(id=train_id, user=request.user)
            model_type = modelo_train.model.model_type.lower()
            model_path = modelo_train.model_path

            labels = self.get_labels(modelo_train)
            modelo_dataset = modelo_train.model
            generated_dataset = modelo_dataset.dataset
            # Usa los tensores del tipo "cnn" si el modelo es svm o naive_bayes
            if model_type in ["svm", "naive_bayes"]:
                folder_path = generated_dataset.folder_paths.get("cnn")
            else:
                folder_path = generated_dataset.folder_paths.get(model_type)

            if not folder_path:
                return Response({"error": "Folder path no encontrado para este tipo de modelo."}, status=status.HTTP_400_BAD_REQUEST)

            tensor, random_file = self.load_random_tensor(folder_path, label)
            self.validate_tensor_shape(tensor, model_type, generated_dataset.data_type)


            tensor = torch.tensor(tensor).unsqueeze(0).float()
            if model_type == "naive_bayes":

                if tensor.ndim == 3:
                    tensor = np.transpose(tensor, (2, 0, 1)) 
                tensor = tensor.reshape(1, -1)


                print("📦 Cargando modelo Naive Bayes...")
                model = joblib.load(model_path)

                start_time = time.time()
                probabilities = model.predict_proba(tensor)[0]
                predicted_index = int(np.argmax(probabilities))
                elapsed_time = (time.time() - start_time) * 1000

            elif model_type == "svm":
                if tensor.ndim == 3:
                    tensor = tensor.squeeze(-1)
                tensor = tensor.reshape(1, -1)

                model = joblib.load(model_path)
                start_time = time.time()
                probabilities = model.predict_proba(tensor)[0]
                predicted_index = int(np.argmax(probabilities))
                elapsed_time = (time.time() - start_time) * 1000

            else:
                model = get_cnn_model(num_classes=len(labels), input_channels=tensor.shape[-1]) if model_type == "cnn" else \
                        get_lstm_model(input_dim=tensor.shape[-1], num_classes=len(labels))
                print("📦 Cargando modelo...")
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                print("🚀 Ejecutando inferencia...")
                start_time = time.time()
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predicted_index = int(np.argmax(probabilities))
                elapsed_time = (time.time() - start_time) * 1000


            print("📊 Probabilidades:", probabilities)
            predicted_class = labels[predicted_index] if predicted_index < len(labels) else str(predicted_index)

            return Response({
                "predicted_class": predicted_class,
                "probabilities": {lbl: float(p) for lbl, p in zip(labels, probabilities)},
                "inference_time_ms": int(elapsed_time),
                "tensor_file": random_file
            })

        except ModeloTrain.DoesNotExist:
            return Response({"error": "Entrenamiento no encontrado."}, status=status.HTTP_400_BAD_REQUEST)
        except FileNotFoundError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": f"Error interno: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetModelMetricsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        train_id = request.query_params.get('train_id')

        try:
            modelo_train = ModeloTrain.objects.get(id=train_id, user=request.user)
            accuracy = modelo_train.accuracy
            loss = modelo_train.loss or 0.0

            combined = modelo_train.combined_data or {}
            confusion_matrix = combined.get("confusion_matrix", [])
            class_names = combined.get("class_names", [f"Clase {i}" for i in range(len(confusion_matrix))])

            return Response({
                "accuracy": accuracy,
                "loss": loss,
                "confusion_matrix": confusion_matrix,
                "class_names": class_names
            })

        except ModeloTrain.DoesNotExist:
            return Response({"error": "Entrenamiento no encontrado."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    


class PredictWithNoiseView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        print("📥 Request recibido en PredictWithNoiseView")

        train_id = request.data.get('train_id')
        noise_level = float(request.data.get('noise_level', 0.05))

        print(f"📌 train_id: {train_id}, noise_level: {noise_level}")

        try:
            modelo_train = ModeloTrain.objects.get(id=train_id, user=request.user)
            model_type = modelo_train.model.model_type.lower()
            model_path = modelo_train.model_path
            print(f"🧠 Modelo: {model_type}, path: {model_path}")

            if model_type in ["svm", "naive_bayes"]:
                labels = modelo_train.combined_data.get("class_names") if modelo_train.combined_data else None
                if not labels:
                    print("❌ No se encontraron labels en combined_data.")
                    return Response({"error": "❌ Labels no encontrados en combined_data."}, status=status.HTTP_400_BAD_REQUEST)
                print(f"✅ Labels desde combined_data: {labels}")
            else:
                model_base, _ = os.path.splitext(model_path)
                model_base = model_base.replace("_model", "")
                label_path = model_base + "_classes.json"
                print(f"🔍 Buscando archivo de clases en: {label_path}")

                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        labels = json.load(f)
                    print(f"✅ Labels cargados: {labels}")
                else:
                    print("❌ Archivo de clases no encontrado.")
                    return Response({"error": "❌ Archivo de clases no encontrado."}, status=status.HTTP_400_BAD_REQUEST)

            modelo_dataset = modelo_train.model
            generated_dataset = modelo_dataset.dataset
            data_type = generated_dataset.data_type
            print(f"📦 Data type: {data_type}")

            # Carpeta según modelo
            if model_type in ["svm", "naive_bayes"]:
                folder_path = generated_dataset.folder_paths.get("cnn")
            else:
                folder_path = generated_dataset.folder_paths.get(model_type)

            print(f"📁 Folder path usado: {folder_path}")

            if not folder_path:
                return Response({"error": "❌ Folder path no encontrado para este tipo."}, status=status.HTTP_400_BAD_REQUEST)

            selected_label = random.choice(labels)
            label_dir = os.path.join(folder_path, selected_label)
            print(f"🎯 Clase seleccionada: {selected_label}, carpeta: {label_dir}")

            files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
            if not files:
                print("❌ No se encontraron tensores para esta clase.")
                return Response({"error": f"No se encontraron tensores para la clase {selected_label}."}, status=status.HTTP_400_BAD_REQUEST)

            random_file = random.choice(files)
            tensor_path = os.path.join(label_dir, random_file)
            print(f"📄 Tensor elegido: {random_file}")

            tensor = np.load(tensor_path)
            print(f"📐 Shape tensor: {tensor.shape}")

            # Validación
            if model_type == "lstm":
                if tensor.ndim != 2 or (data_type == "sensor" and tensor.shape[1] != 9) or (data_type == "audio" and tensor.shape[1] not in [1, 2]):
                    print(f"❌ Tensor inválido para LSTM: {tensor.shape}")
                    return Response({"error": f"❌ Tensor inválido para LSTM: {tensor.shape}"}, status=status.HTTP_400_BAD_REQUEST)

            elif model_type == "cnn":
                if tensor.ndim != 3 or (data_type == "sensor" and tensor.shape[2] != 9) or (data_type == "audio" and tensor.shape[2] not in [1, 2]):
                    print(f"❌ Tensor inválido para CNN: {tensor.shape}")
                    return Response({"error": f"❌ Tensor inválido para CNN: {tensor.shape}"}, status=status.HTTP_400_BAD_REQUEST)

            # Añadir ruido
            noise = np.random.normal(0, noise_level, tensor.shape)
            tensor_noisy = tensor + noise
            tensor_noisy = np.clip(tensor_noisy, 0, 1)
            print(f"🔊 Ruido agregado con nivel: {noise_level}")

            # Inferencia
            if model_type =="svm":

                if tensor_noisy.ndim == 3:
                    tensor_noisy = tensor_noisy.flatten().reshape(1, -1)
                elif tensor_noisy.ndim == 2:
                    tensor_noisy = tensor_noisy.reshape(1, -1)

                print("📦 Cargando modelo sklearn...")
                model = joblib.load(model_path)

                print("🚀 Ejecutando inferencia sklearn...")
                start_time = time.time()
                probabilities = model.predict_proba(tensor_noisy)[0]
                predicted_index = int(np.argmax(probabilities))
                elapsed_time = (time.time() - start_time) * 1000

            elif model_type == "naive_bayes":
                if tensor.ndim == 3:
                    tensor = np.transpose(tensor, (2, 0, 1)) 
                tensor = tensor.reshape(1, -1)

                print(f"🧪 Tensor shape: {tensor.shape}")

                print("📦 Cargando modelo Naive Bayes...")
                model = joblib.load(model_path)

                print("🚀 Ejecutando inferencia Naive Bayes...")
                start_time = time.time()
                probabilities = model.predict_proba(tensor)[0]
                predicted_index = int(np.argmax(probabilities))
                elapsed_time = (time.time() - start_time) * 1000

            else:
                tensor_torch = torch.tensor(tensor_noisy).unsqueeze(0).float()
                if model_type == "cnn":
                    model = get_cnn_model(num_classes=len(labels), input_channels=tensor.shape[-1])
                else:
                    model = get_lstm_model(input_dim=tensor.shape[1], num_classes=len(labels))

                print("📦 Cargando modelo PyTorch...")
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()

                print("🚀 Ejecutando inferencia PyTorch...")
                start_time = time.time()
                with torch.no_grad():
                    output = model(tensor_torch)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predicted_index = int(np.argmax(probabilities))
                elapsed_time = (time.time() - start_time) * 1000

            predicted_class = labels[predicted_index] if predicted_index < len(labels) else str(predicted_index)
            print(f"✅ Clase predicha: {predicted_class}")

            return Response({
                "predicted_class": predicted_class,
                "probabilities": {label: float(prob) for label, prob in zip(labels, probabilities)},
                "inference_time_ms": int(elapsed_time),
                "tensor_file": random_file
            })

        except ModeloTrain.DoesNotExist:
            print("❌ ModeloTrain no encontrado")
            return Response({"error": "❌ Entrenamiento no encontrado."}, status=status.HTTP_400_BAD_REQUEST)
        except ModeloDataset.DoesNotExist:
            print("❌ ModeloDataset no encontrado")
            return Response({"error": "❌ ModeloDataset no encontrado para este usuario."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": f"❌ Error interno: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CompareModelsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        train_id_1 = request.data.get('train_id_1')
        train_id_2 = request.data.get('train_id_2')

        try:
            def predict_with_model(train_id):
                modelo_train = ModeloTrain.objects.get(id=train_id, user=request.user)
                model_type = modelo_train.model.model_type.lower()
                model_path = modelo_train.model_path

                modelo_dataset = modelo_train.model
                generated_dataset = modelo_dataset.dataset
                folder_path = generated_dataset.folder_paths.get(model_type)

                label_path = model_path.replace(".pt", "_classes.json")
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        labels = json.load(f)
                else:
                    raise FileNotFoundError("❌ Archivo de clases no encontrado.")

                selected_label = random.choice(labels)
                label_dir = os.path.join(folder_path, selected_label)
                files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
                random_file = random.choice(files)
                tensor_path = os.path.join(label_dir, random_file)
                tensor = np.load(tensor_path)
                tensor_torch = torch.tensor(tensor).unsqueeze(0).float()

                from core.model_definitions import get_cnn_model, get_lstm_model
                if model_type == "cnn":
                    model = get_cnn_model(num_classes=len(labels), input_channels=tensor.shape[-1])
                else:
                    model = get_lstm_model(input_dim=tensor.shape[1], num_classes=len(labels))

                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()

                start_time = time.time()
                with torch.no_grad():
                    output = model(tensor_torch)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predicted_index = int(np.argmax(probabilities))
                elapsed_time = (time.time() - start_time) * 1000

                predicted_class = labels[predicted_index] if predicted_index < len(labels) else str(predicted_index)
                return {
                    "predicted_class": predicted_class,
                    "inference_time_ms": int(elapsed_time)
                }

            result_1 = predict_with_model(train_id_1)
            result_2 = predict_with_model(train_id_2)

            return Response({
                "model_1": result_1,
                "model_2": result_2
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_user_trainings(request):
    trainings = ModeloTrain.objects.filter(user=request.user).select_related('model')

    data = [
        {
            'id': t.id,
            'training_name': t.training_name,
            'model_type': t.model.model_type,  
            'epochs': t.epochs,
            'accuracy': t.accuracy
        }
        for t in trainings
    ]
    return Response(data)
class ModeloDatasetViewSet(viewsets.ModelViewSet):
    queryset = ModeloDataset.objects.all()
    serializer_class = ModeloDatasetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ModeloDataset.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_dashboard_stats(request):
    user = request.user
    total_models = ModeloTrain.objects.filter(user=user).count()
    total_datasets = GeneratedDataset.objects.filter(user=user).count()
    total_projects = ModeloDataset.objects.filter(user=user).count()

    best_accuracy = ModeloTrain.objects.filter(user=user).aggregate(Max('accuracy'))['accuracy__max'] or 0.0
    best_accuracy = round(best_accuracy * 100, 2)

    return Response({
        "models": total_models,
        "datasets": total_datasets,
        "projects": total_projects,
        "best_accuracy": best_accuracy,
    })

def load_best_tensor(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    best_file = None
    best_std = -1
    for f in files:
        arr = np.load(os.path.join(folder, f))
        std = np.std(arr)
        if std > best_std:
            best_std = std
            best_file = f
    return os.path.join(folder, best_file) if best_file else None


class KMeansClusteringView(APIView):
    def get(self, request):
        dataset_id = request.query_params.get('dataset_id')
        n_clusters = request.query_params.get('n_clusters', 3)
        save_plot = request.query_params.get('plot', 'true').lower() == 'true'

        if not dataset_id:
            return Response({'error': 'Falta el parámetro dataset_id'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            dataset = GeneratedDataset.objects.get(id=dataset_id)
            dataset_folder = dataset.folder_paths.get("cnn")

            if not dataset_folder:
                return Response({'error': 'El dataset no tiene carpeta CNN asociada.'}, status=status.HTTP_400_BAD_REQUEST)

            cluster_labels, plot_path = run_kmeans_clustering(
                dataset_folder=dataset_folder,
                n_clusters=int(n_clusters),
                save_plot=save_plot,
                experiment_name=dataset.name
            )

            return Response({
                'status': 'ok',
                'n_clusters': int(n_clusters),
                'cluster_labels': cluster_labels.tolist(),
                'plot_path': plot_path if save_plot else None
            })

        except GeneratedDataset.DoesNotExist:
            return Response({'error': 'Dataset no encontrado'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
import os, io, base64
from PIL import Image

class LatestYoloImagesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        base_path = '/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/runs/detect/train4'
        image_files = {
            "train_image": "train_batch0.jpg",
            "val_labels_image": "val_batch0_labels.jpg"
        }

        result = {}

        for key, filename in image_files.items():
            image_path = os.path.join(base_path, filename)
            if not os.path.exists(image_path):
                result[key] = None
                continue

            try:
                with Image.open(image_path) as img:
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    encoded = base64.b64encode(buffered.getvalue()).decode()
                    result[key] = encoded
            except Exception as e:
                result[key] = None

        if not result["train_image"] and not result["val_labels_image"]:
            return Response({"error": "No se encontraron imágenes."}, status=404)

        return Response(result)
