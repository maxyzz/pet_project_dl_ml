import psycopg2

import os
from dotenv import load_dotenv
load_dotenv('.env')

def postgres_connection():
    """Устанавливает и возвращает соединение с PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            port=6432,
            database=os.environ['PG_DB'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD'],
        )
    except Exception as e:
        print("❌ Ошибка при подключении к базе данных.")
        raise e

    conn.autocommit = True

    return conn