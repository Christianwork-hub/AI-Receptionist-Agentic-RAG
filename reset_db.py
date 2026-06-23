import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DATABASE_URI")

def reset_db():
    try:
        with psycopg.connect(DB_URI) as conn:
            conn.execute("TRUNCATE TABLE checkpoints CASCADE;")
            conn.execute("TRUNCATE TABLE checkpoint_blobs CASCADE;")
            conn.execute("TRUNCATE TABLE checkpoint_writes CASCADE;")
            conn.commit()
        print("✅ Checkpoints table truncated successfully. Chat history is cleared.")
    except Exception as e:
        print(f"Error clearing db: {e}")

if __name__ == "__main__":
    reset_db()
