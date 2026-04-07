import asyncio
import logging
import time

import numpy as np
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from app.config.Settings import get_settings
from app.service.embedding.EmbeddingFactory import get_embedding_service


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Embedding Helper
# -----------------------------------------------------------------------------
def _get_custom_embedding(texts: list[str]):
    """Generate a Gemini embedding for a given text."""
    embedding_service = get_embedding_service()
    return embedding_service.embed_batch(texts)


def _process_in_batches(cur, data):
    total_rows = len(data)

    for start in range(0, total_rows, get_settings().BATCH_SIZE):
        end = min(start + get_settings().BATCH_SIZE, total_rows)
        batch_df = data.iloc[start:end].copy()
        logging.info(f"Processing rows {start}–{end} of {total_rows}")

        texts_to_embed:  list[str] = batch_df["overall"].tolist()
        embeddings = _get_custom_embedding(texts_to_embed)
        batch_df["embeddings"] = embeddings
        # Prepare tuples for insertion
        data_list = [
            (row["ResumeID"], row["Name"], row["Category"], row["Education"], row["Skills"], row["Summary"],
             np.array(row["embeddings"]))
            for _, row in batch_df.iterrows()
        ]

        # Insert into Postgres
        execute_values(
            cur,
            f"INSERT INTO {get_settings().TABLE_NAME} (resume_id, name, category,education, skills, summary, embedding) VALUES %s",
            data_list,
        )
        logging.info(f"Inserted {len(data_list)} rows into DB")

        # Sleep briefly to prevent 429 rate limit
        time.sleep(get_settings().SLEEP_BETWEEN_BATCHES)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def _do_batch_insert():
    """Embed a slice of the wine review dataset and store it in Postgres."""
    # Connect to PostgreSQL , with pgvector extension
    logging.info(f'Connecting to db with url: {get_settings().DB_DSN}')

    conn = psycopg2.connect(
        get_settings().DB_DSN
    )
    cur = conn.cursor()

    try:
        # Enable pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)

        # Load CSV and subset
        pd_data = pd.read_json(get_settings().csv_file_path, lines=True)
        data = pd_data.iloc[1:48].copy()
        logging.info(f"Processing rows {pd_data.columns} into DB")
        logging.info(f"Total rows selected from CSV: {len(data)}")

        # Create target table
        table_create_sql = f"""
        CREATE TABLE IF NOT EXISTS {get_settings().TABLE_NAME} (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            resume_id TEXT NOT NULL,
            name TEXT,
            category TEXT,
            education TEXT,
            skills TEXT,
            summary TEXT,
            embedding VECTOR(1024)
        );
        """
        logging.info(f'Creating table: {table_create_sql}')
        cur.execute(table_create_sql)
        _process_in_batches(cur, data)
        index_creation_sql = f"""
                CREATE INDEX IF NOT EXISTS {get_settings().TABLE_NAME}_embedding_hnsw_idx ON {get_settings().TABLE_NAME} USING hnsw (embedding vector_cosine_ops);
        """
        logging.info(f'Creating index: {index_creation_sql}')
        cur.execute(index_creation_sql)
    except FileNotFoundError as fnf:
        print(f"CSV not found: {fnf}")

    finally:
        cur.close()
        conn.commit()
        conn.close()
        print("Connection closed.")


def table_exists(table_name: str) -> bool:
    """Return True if table exists, else False."""
    try:
        logging.info(f'Connecting to db with url for table exist test: {get_settings().DB_DSN}')
        conn = psycopg2.connect(
            get_settings().DB_DSN
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);",
            (table_name,),
        )
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False


async def batch_insert_async():
    """Skip if table exists, else perform insert asynchronously."""
    exists = await asyncio.to_thread(table_exists, get_settings().TABLE_NAME)
    if exists:
        logging.info(f"Table '{get_settings().TABLE_NAME}' already exists — skipping insert.")
        return None
    return await asyncio.to_thread(_do_batch_insert)
