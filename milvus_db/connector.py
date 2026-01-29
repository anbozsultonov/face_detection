from pymilvus import connections, db
import os


class MilvusConnector:
    _connected = False

    @classmethod
    def get_connection(cls):
        alias = "default"
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        db_name = os.getenv("MILVUS_DB", "default")  # По умолчанию используем default

        if not cls._connected:
            try:
                # 1. Сначала просто подключаемся к серверу
                connections.connect(alias=alias, host=host, port=port)

                # 2. Если имя базы не default, проверяем её наличие и создаем
                if db_name != "default":
                    existing_dbs = db.list_database()
                    if db_name not in existing_dbs:
                        db.create_database(db_name)
                        print(f"   [Milvus] База данных '{db_name}' создана.")

                # 3. Переключаем контекст соединения на нашу базу
                # Это важно: все последующие операции будут идти в этой БД
                connections.connect(alias=alias, host=host, port=port, db_name=db_name)

                cls._connected = True
                print(f"   [Milvus] Соединение установлено с БД: {db_name}")
            except Exception as e:
                print(f"   [Milvus] Ошибка подключения: {e}")
                raise

        return connections.get_connection(alias)