from deepface import DeepFace

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