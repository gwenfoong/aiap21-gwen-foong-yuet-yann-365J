import sqlite3
import pandas as pd
from . import config


def load_data() -> pd.DataFrame:
    """
    Connects to the SQLite database and loads the main table as a pandas DataFrame.
    """
    conn = sqlite3.connect(config.DB_PATH)
    try:
        query = f"SELECT * FROM {config.TABLE_NAME};"
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.head())