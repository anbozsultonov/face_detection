import cv2
import numpy as np
from insightface.app import FaceAnalysis
from mysql_db.connection import MySQLConnector
from milvus_db.connection import MilvusConnector
from pymilvus import Collection


class FaceService:
    def __init__(self):
        # Инициализируем InsightFace
        # ctx_id=0 если есть GPU (NVIDIA), ctx_id=-1 для CPU (Mac)
        self.app = FaceAnalysis(name='buffalol_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        # Подключаем коллекции
        MilvusConnector.get_connection()
        self.milvus_collection = Collection("face_embeddings")
        self.milvus_collection.load()  # Загружаем коллекцию в память для поиска

    def register_person(self, name, info, image_path):
        # 1. Читаем фото
        img = cv2.imread(image_path)
        if img is None:
            return "Ошибка: Не удалось прочитать файл изображения"

        # 2. Находим лица и извлекаем векторы
        faces = self.app.get(img)
        if len(faces) == 0:
            return "Лица не обнаружены"

        # Берем первое лицо (самое большое)
        face = faces[0]
        embedding = face.normed_embedding.tolist()  # Наш вектор 512

        # 3. Сохраняем в Milvus
        mr = self.milvus_collection.insert([
            [image_path],  # поле path
            [embedding]  # поле embedding
        ])
        person_id = mr.primary_keys[0]  # Получаем ID из Milvus

        # 4. Сохраняем в MySQL
        mysql_conn = MySQLConnector.get_connection()
        cursor = mysql_conn.cursor()
        query = "INSERT INTO people (name, info, person_id) VALUES (%s, %s, %s)"
        cursor.execute(query, (name, info, person_id))
        mysql_conn.commit()
        cursor.close()

        print(f"✅ Человек {name} успешно зарегистрирован с ID: {person_id}")
        return person_id