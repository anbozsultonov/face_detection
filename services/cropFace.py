from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
import cv2


class CropFace:
    @staticmethod
    def detect_and_crop_face(image_path):
        try:
            # Используем mtcnn — он намного лучше находит лица, чем opencv
            detected_faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='mtcnn',  # Меняем с 'opencv' на 'mtcnn'
                enforce_detection=True,
                align=True  # Выравнивает лицо (делает глаза на одной линии)
            )

            # Если лицо найдено, берем первый результат
            face_img = detected_faces[0]["face"]

            # Конвертируем нормализованный массив в формат для OpenCV
            face_bgr = cv2.cvtColor((face_img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
            return face_bgr

        except (FaceNotDetected, ValueError):
            # Если mtcnn не справился, попробуем еще раз с 'opencv' как запасной вариант
            try:
                print("🔄 MTCNN не нашел, пробую через OpenCV...")
                detected_faces = DeepFace.extract_faces(
                    img_path=image_path,
                    detector_backend='opencv',
                    enforce_detection=True
                )
                face_img = detected_faces[0]["face"]
                return cv2.cvtColor((face_img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
            except:
                return None

    @staticmethod
    def detect_all_faces(image_path):
        """Находит все лица на фото и возвращает их массивы и координаты."""
        try:
            detected_faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='mtcnn',
                enforce_detection=True,
                align=True
            )

            results = []
            for face_data in detected_faces:
                face_img = face_data["face"]
                # Конвертируем в формат BGR для OpenCV
                face_bgr = cv2.cvtColor((face_img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

                results.append({
                    "face": face_bgr,
                    "area": face_data["facial_area"]  # x, y, w, h
                })
            return results
        except Exception as e:
            print(f"❌ Ошибка при поиске лиц: {e}")
            return []