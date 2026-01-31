from pymilvus import Collection
from milvus_db import MilvusConnector
from mysql_db import MySQLConnector
from services.fileToEmbedding import FileToEmbedding
from services.cropFace import CropFace
import cv2
from dotenv import load_dotenv
import os
from configs import FACE_RECOGNITION_MILVUS_COLLECTION_NAME

load_dotenv()


class GetInfoByImage:
    def __init__(self, collection_name=FACE_RECOGNITION_MILVUS_COLLECTION_NAME):
        self.milvus_collection = Collection(collection_name)
        self.milvus_collection.load()
        self.storage_base = os.getenv("STORAGE_BASE_PATH", "storage")

    def search(self, face_array, threshold=0.5, limit=3):
        try:
            query_embedding = FileToEmbedding.get_face_embedding(face_array)
            if query_embedding is None:
                return {"status": "error", "message": "Failed to generate embedding"}

            # 2. Поиск в Milvus с использованием RADIUS
            # Для L2: ищем результаты, где distance <= radius
            search_params = {
                "metric_type": "L2",
                "params": {
                    "nprobe": 10,
                    "radius": threshold
                }
            }

            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["path"]
            )

            # Если в заданном радиусе (threshold) ничего не найдено
            if not results or len(results[0]) == 0:
                return {"status": "not_found", "message": f"No matches found within distance {threshold}"}

            # 3. Собираем ID и дистанции найденных (уже отфильтрованных) людей
            hits = results[0]
            person_ids = [str(hit.id) for hit in hits]
            distances = {str(hit.id): hit.distance for hit in hits}

            # 4. Один групповой запрос в MySQL (WHERE IN)
            people_data = self._get_multiple_info_from_mysql(person_ids)

            # 5. Формируем финальный список
            final_results = []
            for person in people_data:
                p_id = str(person['person_id'])
                person_dir = os.path.join(self.storage_base, p_id)
                files = []
                if os.path.exists(person_dir):
                    # Получаем список всех файлов (face и original)
                    files = [os.path.join(person_dir, f) for f in os.listdir(person_dir)
                             if os.path.isfile(os.path.join(person_dir, f))]

                final_results.append({
                    "person_id": p_id,
                    "name": person['name'],
                    "info": person['info'],
                    "distance": round(distances[p_id], 4),
                    "files": files
                })

            # Сортируем от самого похожего к менее похожему
            final_results.sort(key=lambda x: x['distance'])

            return {
                "status": "success",
                "count": len(final_results),
                "matches": final_results
            }

        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            return {"status": "error", "message": str(e)}

    def search_by_path(self, image_path, threshold=0.5, limit=3):
        face_array = CropFace.detect_and_crop_face(image_path=image_path)

        if face_array is None:
            print({"status": "error", "message": "Face not detected"})

        return self.search(
            face_array=face_array,
            threshold=threshold,
            limit=limit
        )

    def identify_group(self, image_path, threshold=5):
        """Находит всех людей на фото и рисует рамки с именами."""
        # Загружаем оригинал для рисования
        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Image not found"}

        # 1. Получаем все лица
        faces_found = CropFace.detect_all_faces(image_path)

        recognized_count = 0

        for item in faces_found:
            face_array = item["face"]
            area = item["area"]

            # 2. Векторизация конкретного лица
            embedding = FileToEmbedding.get_face_embedding(face_array)
            if embedding is None:
                continue

            # 3. Поиск в Milvus (берем только 1 лучшее совпадение)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10, "radius": threshold}}
            milvus_res = self.milvus_collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=1
            )

            name = "Unknown"
            color = (0, 0, 255)  # Красный для незнакомцев

            if milvus_res and len(milvus_res[0]) > 0:
                hit = milvus_res[0][0]
                # Получаем данные из MySQL
                person = self._get_info_from_mysql(hit.id)
                if person:
                    name = person['name']
                    color = (0, 255, 0)  # Зеленый для своих
                    recognized_count += 1

            # 4. Отрисовка
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            # Рамка лица
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Плашка для имени
            cv2.rectangle(img, (x, y - 30), (x + w, y), color, -1)
            # Текст имени
            cv2.putText(img, name, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 5. Сохраняем результат в корень проекта
        output_name = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(output_name, img)

        return {
            "status": "success",
            "detected_faces": len(faces_found),
            "recognized_faces": recognized_count,
            "output_path": output_name
        }

    def _get_multiple_info_from_mysql(self, person_ids):
        """Групповой запрос для Singleton-соединения."""
        if not person_ids: return []
        try:
            mysql_conn = MySQLConnector.get_connection()
            cursor = mysql_conn.cursor(dictionary=True, buffered=True)

            placeholders = ', '.join(['%s'] * len(person_ids))
            query = f"SELECT person_id, name, info FROM people WHERE person_id IN ({placeholders})"

            cursor.execute(query, tuple(person_ids))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"❌ Ошибка MySQL: {e}")
            return []

    def _get_info_from_mysql(self, person_id):
        try:
            mysql_conn = MySQLConnector.get_connection()

            cursor = mysql_conn.cursor(dictionary=True, buffered=True)

            query = "SELECT person_id, name, info FROM people WHERE person_id = %s"

            cursor.execute(query, (person_id,))

            results = cursor.fetchone()
            cursor.close()
            return results
        except Exception as e:
            print(f"❌ Ошибка MySQL: {e}")
            return None

    def identify_group_by_face_recognition(self, image_path, threshold=0.6):  # Увеличили порог до 0.6
        """
                Находит всех людей на фото, используя face_recognition (128 dim).
                Рисует рамки и подписывает имена.
                """
        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Image not found"}

        # Убеждаемся, что коллекция загружена
        self.milvus_collection.load()

        # 1. Находим все лица (используем retinaface для групповых фото)
        faces_found = CropFace.detect_all_faces(image_path)
        recognized_count = 0

        print(f"🔍 Найдено лиц на фото: {len(faces_found)}")

        for i, item in enumerate(faces_found):
            face_array = item["face"]
            area = item["area"]

            # 2. Векторизация через НОВЫЙ метод (128 dim)
            embedding = FileToEmbedding.get_embedding_face_recognition(face_array)
            if embedding is None:
                continue

            # 3. Поиск в Milvus (используем метрику L2)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            milvus_res = self.milvus_collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=1
            )

            name = "Unknown"
            color = (0, 0, 255)  # Красный для незнакомых

            if milvus_res and len(milvus_res[0]) > 0:
                hit = milvus_res[0][0]
                # Логируем для отладки
                print(f"👤 Лицо #{i + 1}: Ближайший ID: {hit.id}, Дистанция: {hit.distance:.4f}")

                # Теперь threshold 0.6 — это стандарт для нормализованного L2
                if hit.distance <= threshold:
                    person = self._get_info_from_mysql(hit.id)
                    if person:
                        name = person['name']
                        color = (0, 255, 0)  # Зеленый для своих
                        recognized_count += 1

            # 4. Рисуем результат
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Добавляем подложку под текст для читаемости
            cv2.rectangle(img, (x, y - 25), (x + w, y), color, -1)
            cv2.putText(img, name, (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Сохраняем результат
        output_path = f"identified_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, img)

        return {
            "status": "success",
            "detected": len(faces_found),
            "recognized": recognized_count,
            "output_path": output_path
        }