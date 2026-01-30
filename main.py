from services.faceRegister import FaceRegisterService
import os
from milvus_db import MilvusConnector
from pymilvus import Collection, connections

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
            name="Anbozsultonov",
            info="Разработчик системы",
            image_path=image_path
        )

        if isinstance(person_id, int) or (isinstance(person_id, str) and person_id.isdigit()):
            print(f"✅ Готово! ID в базе: {person_id}")
        else:
            print(f"❌ Ошибка: {person_id}")
    else:
        print(f"⚠️ Файл {image_path} не найден. Положи фото в папку data/")


def check_milvus_data():
    MilvusConnector.get_connection()
    collection = Collection("face_embeddings")
    collection.load()
    print("Названия полей в твоей коллекции:")
    for field in collection.schema.fields:
        print(f" - {field.name} ({field.dtype})")


def get_milvus_rows():
    # Гарантируем подключение через твой Singleton
    MilvusConnector.get_connection()

    collection = Collection("face_embeddings")
    collection.load()  # Обязательно загружаем в RAM

    # Выбираем записи, где id больше или равен 0
    # В output_fields указываем те имена полей, что выдала схема
    results = collection.query(
        expr="id >= 0",
        output_fields=["id", "path"],
        limit=10
    )

    print(f"🔎 Найдено записей: {len(results)}")
    for row in results:
        print(f"ID: {row['id']} | Путь к фото: {row['path']}")

    return results

if __name__ == "__main__":
    main()
