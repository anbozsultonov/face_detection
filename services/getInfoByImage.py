from pymilvus import Collection
from milvus_db import MilvusConnector
from mysql_db import MySQLConnector
from services.fileToEmbedding import FileToEmbedding
from services.cropFace import CropFace

from dotenv import load_dotenv
import os

load_dotenv()


class GetInfoByImage:
    def __init__(self):
        MilvusConnector.get_connection()
        self.milvus_collection = Collection("face_embeddings")
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
