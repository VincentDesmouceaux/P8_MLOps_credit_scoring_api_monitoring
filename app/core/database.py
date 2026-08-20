import os

import psycopg


def get_database_connection():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "La variable d'environnement DATABASE_URL est absente."
        )

    return psycopg.connect(database_url)