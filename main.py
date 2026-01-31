from services.faceRegister import FaceRegisterService
import os
from milvus_db import MilvusConnector
from pymilvus import Collection, connections
from services import CropFace
from services import FileToEmbedding
from services import SaveImagesToStorage

def main():
    # 1. Инициализация сервиса
    # При первом запуске он скачает модели (~200MB) в ~/.insightface/models/
    print("🚀 Инициализация нейросети...")
    face_app = FaceRegisterService()

    # 2. Регистрация нового человека
    image_path = "photo.jpg"  # Укажи путь к своему фото

    if os.path.exists(image_path):
        print(f"📸 Регистрация человека по фото: {image_path}")

        person_id = face_app.register_person(
            name="Tony stark",
            info="Iron man",
            image_path=image_path
        )

        if isinstance(person_id, int) or (isinstance(person_id, str) and person_id.isdigit()):
            print(f"✅ Готово! ID в базе: {person_id}")
        else:
            print(f"❌ Ошибка: {person_id}")
    else:
        print(f"⚠️ Файл {image_path} не найден. Положи фото в папку data/")

# if __name__ == "__main__":
#     main()

from services.getInfoByImage import GetInfoByImage
#
# app = GetInfoByImage()
# k = 0
# image_path = "many_people.jpg"
# faces_found = CropFace.detect_all_faces(image_path)
# for item in faces_found:
#     cropped_face = item["face"]
#     embedding = FileToEmbedding.get_face_embedding(cropped_face)
#     k+=1
#     SaveImagesToStorage.save_images_to_storage(k, image_path, cropped_face)

# search_service = GetInfoByImage()
# res = search_service.identify_group_by_face_recognition("avengers.jpg", threshold=0.8)
#
# if res["status"] == "success":
#     print(f"✅ Найдено лиц: {res['detected_faces']}, Узнано: {res['recognized_faces']}")
#     print(f"🖼 Результат сохранен в: {res['output_path']}")

register = FaceRegisterService()
register.register_person_by_face_recognition("Тони Старк", "Iron Man", "photo.jpg")
