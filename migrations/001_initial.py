from mysql_db import MySQLConnector
from milvus_db import MilvusConnector
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection


def run():
    #--- 1. Настройка MySQL ---
    mysql_conn = MySQLConnector.get_connection()
    cursor = mysql_conn.cursor()

    # Создаем таблицу people
    # person_id здесь — это ID, который придет из Milvus
    mysql_query = """
    CREATE TABLE IF NOT EXISTS people (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(191) NOT NULL,
        info TEXT,
        person_id VARCHAR(191) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cursor.execute(mysql_query)
    mysql_conn.commit()
    cursor.close()
    print("   [MySQL] Таблица 'people' успешно создана.")

    # --- 2. Настройка Milvus ---
    # В Milvus соединение уже установлено через наш коннектор
    MilvusConnector.get_connection()

    collection_name = "face_embeddings"

    # Описываем поля коллекции
    fields = [
        # Milvus не поддерживает UNSIGNED напрямую, используем INT64
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
        # dim=512 для стандартных моделей InsightFace
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]

    schema = CollectionSchema(fields, description="Коллекция для хранения векторов лиц")

    # Создаем коллекцию, если её нет
    from pymilvus import utility
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)

        # Сразу создаем индекс для быстрого поиска (IVF_FLAT подходит для начала)
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"   [Milvus] Коллекция '{collection_name}' и индекс созданы.")
    else:
        print(f"   [Milvus] Коллекция '{collection_name}' уже существует.")

    collection = Collection("face_embeddings")

    # Параметры индекса
    index_params = {
        "metric_type": "L2",  # Тип метрики (L2 для Facenet512)
        "index_type": "IVF_FLAT",  # Тип индекса
        "params": {"nlist": 128}  # Количество кластеров
    }

    print("⏳ Создание индекса на колонке 'embedding'...")
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print("✅ Индекс успешно создан!")

if __name__ == "__main__":
    run()