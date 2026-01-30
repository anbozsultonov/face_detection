import mysql.connector
import os


class MySQLConnector:
    _connection = None

    @classmethod
    def get_connection(cls):
        if cls._connection is None or not cls._connection.is_connected():
            # Данные берем из окружения
            host = os.getenv("DB_HOST", "127.0.0.1")  # Принудительно 127.0.0.1
            user = os.getenv("DB_USER", "root")
            password = os.getenv("DB_PASS", "root_password")
            db_name = os.getenv("DB_NAME", "face_db")

            try:
                # ШАГ 1: Подключаемся к серверу БЕЗ указания базы данных
                temp_conn = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    auth_plugin='mysql_native_password'
                )

                cursor = temp_conn.cursor()

                # ШАГ 2: Создаем базу данных, если её нет
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                cursor.close()
                temp_conn.close()

                # ШАГ 3: Теперь подключаемся уже к конкретной базе
                cls._connection = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=db_name,
                    charset='utf8mb4'
                )
                print(f"   [MySQL] База '{db_name}' готова, соединение установлено.")

            except mysql.connector.Error as err:
                print(f"   [MySQL] Критическая ошибка: {err}")
                raise
        return cls._connection