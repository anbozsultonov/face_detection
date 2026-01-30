import os
from pymilvus import connections, db


class MilvusConnector:
    _instance = None
    _connected = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusConnector, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_connection(cls):
        alias = "default"
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        db_name = os.getenv("MILVUS_DB", "face_recognition_db")

        if not cls._connected:
            try:
                # Подключаемся
                connections.connect(alias=alias, host=host, port=port)

                if db_name != "default":
                    if db_name not in db.list_database():
                        db.create_database(db_name)
                    connections.disconnect(alias)
                    connections.connect(alias=alias, host=host, port=port, db_name=db_name)

                cls._connected = True
                print(f"   [Milvus] Singleton: Соединение установлено с БД: {db_name}")
            except Exception as e:
                cls._connected = False
                print(f"   [Milvus] Ошибка Singleton: {e}")
                raise

        # Возвращаем алиас (строку). Это стандартный путь для pymilvus.
        return alias