import duckdb
import os
import time
import logging
from minio import Minio
from minio.error import S3Error

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration MinIO
MINIO_ENDPOINT = 'localhost:9000'
MINIO_ACCESS_KEY = 'XHwfOkXcYQqmJZxgo0tK'
MINIO_SECRET_KEY = 'KnQgSiKPJJgmTnk6Pb6Ao5taBV23oPMz63aM9p3l'
MINIO_BUCKET = 'btc'
MINIO_PREFIX = 'BTCUSDT-trades-'

# Configuration DuckDB
DB_PATH = "data/trades.duckdb"
TEMP_DIR = "data/temp"

# Liste des fichiers déjà traités
processed_files = set()

def init_duckdb():
    """Initialise la connexion DuckDB et crée les tables nécessaires"""
    # Créer le répertoire de données si nécessaire
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Créer et configurer la base de données DuckDB
    logger.info("Initialisation de la base de données DuckDB...")
    con = duckdb.connect(DB_PATH)

    # Configurer les paramètres pour optimiser les performances
    con.execute("PRAGMA memory_limit='15GB'")
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{TEMP_DIR}'")

    # Configurer la connexion à MinIO (S3)
    logger.info("Configuration de la connexion à MinIO...")
    con.execute("""
        INSTALL httpfs;
        LOAD httpfs;
        SET s3_region='eu-east-1';
        SET s3_url_style='path';
        SET s3_endpoint='localhost:9000';
        SET s3_access_key_id='XHwfOkXcYQqmJZxgo0tK';
        SET s3_secret_access_key='KnQgSiKPJJgTMnk6Pb6Ao5taBV23oPMz63aM9p3l';
        SET s3_use_ssl=false;
    """)

    # Créer la table principale si elle n'existe pas déjà
    con.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id BIGINT PRIMARY KEY,
        price DOUBLE,
        volume DOUBLE,
        quote_volume DOUBLE,
        timestamp BIGINT,
        is_buyer_maker BOOLEAN,
        is_best_match BOOLEAN
    )
    """)
    
    # Créer la table d'agrégation si elle n'existe pas déjà
    con.execute("""
    CREATE TABLE IF NOT EXISTS trades_aggregated_min (
        timestamp_min BIGINT PRIMARY KEY,
        max_price DOUBLE,
        min_price DOUBLE,
        open_price DOUBLE,
        close_price DOUBLE,
        total_volume DOUBLE,
        quote_volume DOUBLE,
        trade_count INTEGER,
        price_range DOUBLE,
        volatility_pct DOUBLE
    )
    """)
    
    # Créer table de suivi des fichiers traités
    con.execute("""
    CREATE TABLE IF NOT EXISTS processed_files (
        filename VARCHAR PRIMARY KEY,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Récupérer la liste des fichiers déjà traités
    result = con.execute("SELECT filename FROM processed_files").fetchall()
    global processed_files
    processed_files = {row[0] for row in result}
    
    return con

def get_minio_client():
    """Crée et renvoie un client MinIO"""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

def scan_minio_for_new_files():
    """Vérifie s'il y a de nouveaux fichiers dans MinIO"""
    try:
        minio_client = get_minio_client()
        objects = minio_client.list_objects(MINIO_BUCKET, prefix=MINIO_PREFIX, recursive=True)
        
        new_files = []
        for obj in objects:
            if obj.object_name not in processed_files:
                new_files.append(f's3://{MINIO_BUCKET}/{obj.object_name}')
        
        return new_files
    
    except S3Error as e:
        logger.error(f"Erreur lors du scan de MinIO: {e}")
        return []

def process_file(con, file_path):
    """Traite un fichier en l'important dans DuckDB et en l'agrégeant"""
    logger.info(f"Traitement du fichier: {file_path}")
    
    try:
        # Importer les données
        con.execute(f"""
        COPY trades FROM '{file_path}' (
            HEADER FALSE,
            DELIMITER ','
        )
        """)
        
        # Agréger les données
        logger.info("Agrégation des nouvelles données...")
        con.execute("""
            INSERT INTO trades_aggregated_min
            WITH normalized_trades AS (
                SELECT 
                    *,
                    CASE 
                        WHEN LENGTH(CAST(timestamp AS VARCHAR)) > 13 
                        THEN timestamp / POWER(10, LENGTH(CAST(timestamp AS VARCHAR)) - 13)
                        ELSE timestamp
                    END AS normalized_timestamp
                FROM trades
            ),
            ranked_trades AS (
                SELECT 
                    *,
                    CAST(normalized_timestamp / (1000*60) AS BIGINT) as timestamp_min,
                    ROW_NUMBER() OVER (PARTITION BY CAST(normalized_timestamp / (1000*60) AS BIGINT) ORDER BY normalized_timestamp ASC) as rn_first,
                    ROW_NUMBER() OVER (PARTITION BY CAST(normalized_timestamp / (1000*60) AS BIGINT) ORDER BY normalized_timestamp DESC) as rn_last
                FROM normalized_trades
            ),
            open_close_prices AS (
                SELECT 
                    timestamp_min,
                    MAX(CASE WHEN rn_first = 1 THEN price END) as open_price,
                    MAX(CASE WHEN rn_last = 1 THEN price END) as close_price
                FROM ranked_trades
                GROUP BY timestamp_min
            )
            SELECT 
                r.timestamp_min,
                MAX(r.price) as max_price,
                MIN(r.price) as min_price,
                oc.open_price,
                oc.close_price,
                SUM(r.volume) as total_volume,
                SUM(r.price * r.volume) as quote_volume,
                COUNT(*) as trade_count,
                MAX(r.price) - MIN(r.price) as price_range,
                (MAX(r.price) - MIN(r.price)) / NULLIF(MIN(r.price), 0) * 100 as volatility_pct
            FROM ranked_trades r
            JOIN open_close_prices oc ON r.timestamp_min = oc.timestamp_min
            GROUP BY r.timestamp_min, oc.open_price, oc.close_price
            ORDER BY r.timestamp_min
            ON CONFLICT (timestamp_min) DO UPDATE SET
                max_price = GREATEST(excluded.max_price, trades_aggregated_min.max_price),
                min_price = LEAST(excluded.min_price, trades_aggregated_min.min_price),
                total_volume = trades_aggregated_min.total_volume + excluded.total_volume,
                quote_volume = trades_aggregated_min.quote_volume + excluded.quote_volume,
                trade_count = trades_aggregated_min.trade_count + excluded.trade_count,
                price_range = GREATEST(excluded.max_price, trades_aggregated_min.max_price) - LEAST(excluded.min_price, trades_aggregated_min.min_price),
                volatility_pct = (GREATEST(excluded.max_price, trades_aggregated_min.max_price) - LEAST(excluded.min_price, trades_aggregated_min.min_price)) / NULLIF(LEAST(excluded.min_price, trades_aggregated_min.min_price), 0) * 100
        """)
        
        # Marquer le fichier comme traité
        filename = file_path.split('/')[-1]
        con.execute(f"INSERT INTO processed_files (filename) VALUES ('{filename}')")
        processed_files.add(filename)
        
        logger.info(f"Fichier {filename} traité avec succès")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du fichier {file_path}: {e}")
        return False

def main():
    """Fonction principale qui surveille MinIO et traite les nouveaux fichiers"""
    logger.info("Démarrage du service de surveillance de MinIO")
    
    con = init_duckdb()
    
    try:
        while True:
            # Vérifier les nouveaux fichiers
            new_files = scan_minio_for_new_files()
            
            if new_files:
                logger.info(f"Détection de {len(new_files)} nouveaux fichiers à traiter")
                
                for file_path in new_files:
                    success = process_file(con, file_path)
                    if success:
                        logger.info(f"Traitement de {file_path} terminé")
                    else:
                        logger.error(f"Échec du traitement de {file_path}")
                
                # Optimiser la base après un lot de traitements
                logger.info("Optimisation de la base de données...")
                con.execute("VACUUM")
                con.execute("ANALYZE")
            else:
                logger.info("Aucun nouveau fichier détecté")
            
            # Attendre avant la prochaine vérification
            time.sleep(60)  # Vérifie toutes les minutes
    
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    finally:
        con.close()
        logger.info("Connexion à la base de données fermée")

if __name__ == "__main__":
    main()



