from milvus_db import MilvusConnector
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility


def run():
    # Инициализируем соединение
    MilvusConnector.get_connection()

    collection_name = "people_fr"  # Новая коллекция для face-recognition

    if utility.has_collection(collection_name):
        print(f"⚠️ Коллекция '{collection_name}' уже существует. Миграция не требуется.")
        return

    # Определяем схему для векторов размерностью 128
    fields = [
        FieldSchema(name="person_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)  # Для face_recognition
    ]

    schema = CollectionSchema(fields, "Collection for 128-dim face embeddings")
    collection = Collection(collection_name, schema)

    # Создаем индекс L2 (евклидово расстояние)
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    print(f"✅ Миграция завершена! Создана коллекция '{collection_name}' с dim=128.")


if __name__ == "__main__":
    run()