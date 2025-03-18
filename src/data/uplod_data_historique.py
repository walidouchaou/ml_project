"""Configuration et chargement des données historiques de trading.

Ce module gère l'importation des données historiques de trading depuis MinIO vers DuckDB.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import duckdb
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MinIOConfig:
    """Configuration pour la connexion MinIO."""
    endpoint: str
    access_key: str
    secret_key: str
    region: str = 'eu-east-1'
    use_ssl: bool = False
    url_style: str = 'path'

@dataclass
class DuckDBConfig:
    """Configuration pour DuckDB."""
    db_path: str
    memory_limit: str = '15GB'
    threads: int = 8
    temp_directory: str = 'data/temp'

class TradeDataLoader:
    """Gestionnaire de chargement des données de trading."""
    
    def __init__(
        self, 
        minio_config: MinIOConfig,
        db_config: DuckDBConfig
    ):
        self.minio_config = minio_config
        self.db_config = db_config
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def _init_database(self) -> None:
        """Initialise la connexion DuckDB et configure les paramètres."""
        logger.info("Initialisation de la base de données DuckDB")
        os.makedirs(os.path.dirname(self.db_config.db_path), exist_ok=True)
        
        self.conn = duckdb.connect(self.db_config.db_path)
        self.conn.execute(f"PRAGMA memory_limit='{self.db_config.memory_limit}'")
        self.conn.execute(f"PRAGMA threads={self.db_config.threads}")
        self.conn.execute(f"PRAGMA temp_directory='{self.db_config.temp_directory}'")

    def _configure_minio(self) -> None:
        """Configure la connexion MinIO."""
        logger.info("Configuration de la connexion MinIO")
        self.conn.execute("""
            INSTALL httpfs;
            LOAD httpfs;
        """)
        self.conn.execute(f"SET s3_region='{self.minio_config.region}'")
        self.conn.execute(f"SET s3_url_style='{self.minio_config.url_style}'")
        self.conn.execute(f"SET s3_endpoint='{self.minio_config.endpoint}'")
        self.conn.execute(f"SET s3_access_key_id='{self.minio_config.access_key}'")
        self.conn.execute(f"SET s3_secret_access_key='{self.minio_config.secret_key}'")
        self.conn.execute(f"SET s3_use_ssl={str(self.minio_config.use_ssl).lower()}")

    def _create_tables(self) -> None:
        """Crée les tables nécessaires."""
        logger.info("Création des tables")
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id BIGINT PRIMARY KEY,
            price DOUBLE,
            volume DOUBLE,
            quote_volume DOUBLE,
            timestamp BIGINT,
            is_buyer_maker BOOLEAN,
            is_best_match BOOLEAN
        );
        
        CREATE TABLE IF NOT EXISTS trades_aggregated_min (
            timestamp_min BIGINT,
            max_price DOUBLE,
            min_price DOUBLE,
            open_price DOUBLE,
            close_price DOUBLE,
            total_volume DOUBLE,
            quote_volume DOUBLE,
            trade_count BIGINT,
            price_range DOUBLE,
            volatility_pct DOUBLE
        );
        """)

    def process_file(self, file_path: str) -> None:
        """Traite un fichier CSV individuel."""
        try:
            logger.info(f"Traitement du fichier: {file_path}")
            
            # Import des données
            self.conn.execute(f"""
            COPY trades FROM '{file_path}' (
                HEADER FALSE,
                DELIMITER ','
            )
            """)
            
            # Agrégation des données
            self._aggregate_data()
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {file_path}: {str(e)}")
            raise

    def _aggregate_data(self) -> None:
        """Agrège les données par minute."""
        logger.info("Agrégation des données")
        # ... [Le reste de votre requête SQL d'agrégation reste identique] ...

    def process_files(self, file_paths: List[str]) -> None:
        """Traite une liste de fichiers CSV."""
        try:
            self._init_database()
            self._configure_minio()
            self._create_tables()

            for i, file_path in enumerate(file_paths, 1):
                logger.info(f"Traitement du fichier {i}/{len(file_paths)}")
                self.process_file(file_path)
                
            self._optimize_database()
            
        finally:
            if self.conn:
                self.conn.close()

    def _optimize_database(self) -> None:
        """Optimise la base de données."""
        logger.info("Optimisation de la base de données")
        self.conn.execute("VACUUM")
        self.conn.execute("ANALYZE")

def main():
    """Point d'entrée principal."""
    load_dotenv()  # Charge les variables d'environnement depuis .env

    minio_config = MinIOConfig(
        endpoint=os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
        access_key=os.getenv('MINIO_ACCESS_KEY', 'XHwfOkXcYQqmJZxgo0tK'),
        secret_key=os.getenv('MINIO_SECRET_KEY', 'KnQgSiKPJJgTMnk6Pb6Ao5taBV23oPMz63aM9p3l')
    )

    db_config = DuckDBConfig(
        db_path="data/trades.duckdb"
    )

    fichiers_csv = [
        's3://btc/BTCUSDT-trades-2022-01.csv',
        's3://btc/BTCUSDT-trades-2022-02.csv',
        's3://btc/BTCUSDT-trades-2022-03.csv',
        's3://btc/BTCUSDT-trades-2022-04.csv',
        's3://btc/BTCUSDT-trades-2022-05.csv',
        's3://btc/BTCUSDT-trades-2022-06.csv',
        's3://btc/BTCUSDT-trades-2022-07.csv',
        's3://btc/BTCUSDT-trades-2022-08.csv',
        's3://btc/BTCUSDT-trades-2022-09.csv',
        's3://btc/BTCUSDT-trades-2022-10.csv',
        's3://btc/BTCUSDT-trades-2022-11.csv',
        's3://btc/BTCUSDT-trades-2022-12.csv',
        's3://btc/BTCUSDT-trades-2021-01.csv',
        's3://btc/BTCUSDT-trades-2021-02.csv',
        's3://btc/BTCUSDT-trades-2021-03.csv',
        's3://btc/BTCUSDT-trades-2021-04.csv',
        's3://btc/BTCUSDT-trades-2021-05.csv',
        's3://btc/BTCUSDT-trades-2021-06.csv',
        's3://btc/BTCUSDT-trades-2021-07.csv',
        's3://btc/BTCUSDT-trades-2021-09.csv',
        's3://btc/BTCUSDT-trades-2021-10.csv',
        's3://btc/BTCUSDT-trades-2021-11.csv',
        's3://btc/BTCUSDT-trades-2021-12.csv'
    ]

    loader = TradeDataLoader(minio_config, db_config)
    loader.process_files(fichiers_csv)

if __name__ == "__main__":
    main()



