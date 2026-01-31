from deepface import DeepFace
import cv2
import face_recognition
import numpy as np

class FileToEmbedding:
    @staticmethod
    def get_embedding(image_path):
        """Извлекает вектор лица (Facenet512 + RetinaFace)."""
        objs = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=True
        )
        if not objs:
            raise ValueError("Лицо не обнаружено")
        return objs[0]["embedding"]


    @staticmethod
    def get_face_embedding(face_array):
        try:
            results = DeepFace.represent(
                img_path=face_array,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend='skip',
                align=False
            )
            return results[0]["embedding"]

        except Exception as e:
            print(f"❌ Ошибка векторизации: {e}")
            return None

    @staticmethod
    def get_embedding_face_recognition(face_array):
        """
        НОВЫЙ МЕТОД: Использует face_recognition (128 dim).
        Ожидает вырезанное лицо (BGR массив от OpenCV).
        """
        try:
            # Превращаем BGR (OpenCV) в RGB (face_recognition)
            rgb_face = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)

            # Так как лицо уже вырезано, передаем всё изображение как одну локацию
            height, width, _ = rgb_face.shape
            face_location = [(0, width, height, 0)]

            # Извлекаем 128-мерный вектор
            encodings = face_recognition.face_encodings(rgb_face, known_face_locations=face_location)

            if len(encodings) > 0:
                embedding = encodings[0]

                # L2 Нормализация для Milvus
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                return embedding.tolist()
            return None
        except Exception as e:
            print(f"❌ Ошибка face_recognition: {e}")
            return None