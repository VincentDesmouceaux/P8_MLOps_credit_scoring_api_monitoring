import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Charge .env pour le développement local.
# Une variable déjà définie dans l'environnement, notamment sur Render,
# n'est pas remplacée grâce à override=False.
load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# -------------------------------------------------------------------
# URL PostgreSQL
# -------------------------------------------------------------------

def get_database_url() -> str:
    """
    Retourne l'URL de connexion PostgreSQL.

    En production, DATABASE_URL est fournie par l'environnement
    d'exécution (Render).

    En développement local, elle peut être chargée depuis
    le fichier .env situé à la racine du projet.
    """
    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise RuntimeError(
            "La variable d'environnement DATABASE_URL "
            "est absente. Configurez-la dans Render ou "
            "dans le fichier .env pour le développement local."
        )

    return database_url


# -------------------------------------------------------------------
# Connexion PostgreSQL
# -------------------------------------------------------------------

def get_database_connection():
    """
    Crée et retourne une connexion PostgreSQL avec psycopg.

    La connexion est créée à la demande afin d'éviter
    de maintenir une connexion globale ouverte.
    """
    database_url = get_database_url()

    return psycopg.connect(
        database_url
    )