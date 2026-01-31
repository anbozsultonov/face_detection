from pymilvus import Collection

from configs import FACE_RECOGNITION_MILVUS_COLLECTION_NAME
# Импорты без .connector
from milvus_db import MilvusConnector
from mysql_db import MySQLConnector
from services import FileToEmbedding, CropFace, SaveImagesToStorage


class FaceRegisterService:
    def __init__(self, collection_name=FACE_RECOGNITION_MILVUS_COLLECTION_NAME):
        # Инициализация Milvus через Singleton
        MilvusConnector.get_connection()
        self.milvus_collection = Collection(collection_name)
        self.milvus_collection.load()

    def insert_to_milvus(self, embedding, original_path):
        """Записывает вектор в Milvus и возвращает строковый ID."""
        if embedding is None or len(embedding) != 128:
            print(f"❌ Ошибка: Вектор имеет неверный размер ({len(embedding) if embedding else 0})")
            return None

            # 2. Убеждаемся, что person_id — это целое число (INT64)
        try:
            p_id = int(person_id)
        except ValueError:
            print(f"❌ Ошибка: person_id должен быть числом, а не {person_id}")
            return None

        mr = self.milvus_collection.insert([
            [original_path],
            [embedding]
        ])
        self.milvus_collection.flush()
        return str(mr.primary_keys[0])



    def save_to_mysql(self, name, info, person_id):
        """Записывает метаданные в MySQL (UTF-8)."""
        mysql_conn = MySQLConnector.get_connection()
        cursor = mysql_conn.cursor()
        query = "INSERT INTO people (name, info, person_id) VALUES (%s, %s, %s)"
        cursor.execute(query, (name, info, person_id))
        mysql_conn.commit()
        cursor.close()

    def register_person(self, name, info, image_path):
        """Полный публичный цикл регистрации."""
        try:
            cropped_face = CropFace.detect_and_crop_face(image_path)
            if cropped_face is None:
                print("Лицо не найдено, прерываем регистрацию...")
                return None

            embedding = FileToEmbedding.get_face_embedding(cropped_face)

            person_id = self.insert_to_milvus(embedding, image_path)
            SaveImagesToStorage.save_images_to_storage(person_id, image_path, cropped_face)
            self.save_to_mysql(name, info, person_id)

            print(f"✅ Успешно зарегистрирован: {name} (ID: {person_id})")

            return person_id
        except Exception as e:
            print(f"❌ Ошибка в FaceRegisterService: {e}")
            raise

    def register_person_by_face_recognition(self, name, info, image_path):
        """Полный публичный цикл регистрации."""
        try:
            cropped_face = CropFace.detect_and_crop_face(image_path)
            if cropped_face is None:
                print("Лицо не найдено, прерываем регистрацию...")
                return None

            embedding = FileToEmbedding.get_embedding_face_recognition(cropped_face)

            person_id = self.insert_to_milvus(embedding, image_path)
            SaveImagesToStorage.save_images_to_storage(person_id, image_path, cropped_face)
            self.save_to_mysql(name, info, person_id)

            print(f"✅ Успешно зарегистрирован: {name} (ID: {person_id})")

            return person_id
        except Exception as e:
            print(f"❌ Ошибка в FaceRegisterService: {e}")
            raise
