from pymilvus import connections, db
import os


class MilvusConnector:
    _connected = False

    @classmethod
    def get_connection(cls):
        alias = "default"
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        db_name = os.getenv("MILVUS_DB", "default")

        if not cls._connected:
            try:
                # 1. Подключаемся к серверу
                connections.connect(alias=alias, host=host, port=port)

                # 2. Создаем БД если нужно
                if db_name != "default":
                    if db_name not in db.list_database():
                        db.create_database(db_name)

                # 3. Переподключаемся к конкретной БД
                connections.disconnect(alias)  # Переподключаемся для смены БД
                connections.connect(alias=alias, host=host, port=port, db_name=db_name)

                cls._connected = True
                print(f"   [Milvus] Соединение установлено с БД: {db_name}")
            except Exception as e:
                print(f"   [Milvus] Ошибка: {e}")
                raise

        # Просто возвращаем True или alias, так как pymilvus работает глобально
        # через установленное соединение "default"
        return alias