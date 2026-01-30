import os
import cv2
import shutil
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

class SaveImagesToStorage:
    @staticmethod
    def save_images_to_storage(person_id, original_source, face_array):
        storage_base = os.getenv("STORAGE_BASE_PATH", "storage")
        person_dir = os.path.join(storage_base, str(person_id))
        os.makedirs(person_dir, exist_ok=True)
        photo_name, ext = os.path.splitext(os.path.basename(original_source))

        orig_dest = os.path.join(person_dir, f"{person_id}.original.{photo_name}{ext}")
        face_dest = os.path.join(person_dir, f"{person_id}.face.{photo_name}{ext}")

        try:
            shutil.copy2(original_source, orig_dest)
            cv2.imwrite(face_dest, face_array)

            return orig_dest, face_dest

        except Exception as e:
            print(f"❌ Static Save Error: {e}")
            raise