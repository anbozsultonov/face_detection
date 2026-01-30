from mysql_db import MySQLConnector
from milvus_db import MilvusConnector
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection


def run():
    # --- 1. Настройка MySQL ---
    mysql_conn = MySQLConnector.get_connection()
    cursor = mysql_conn.cursor()

    # Создаем таблицу people
    # person_id здесь — это ID, который придет из Milvus
    mysql_query = """
    ALTER TABLE people CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    
    ALTER TABLE people 
        MODIFY person_id VARCHAR(255),
        ADD INDEX idx_person_id (person_id);
    """
    cursor.execute(mysql_query)
    mysql_conn.commit()
    cursor.close()
    print("   [MySQL] person_id in people changes successfully.")

if __name__ == "__main__":
    run()