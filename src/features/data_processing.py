import pandas as pd
import os
import json
from typing import Optional, Generator
from datetime import datetime
import time
import logging
import duckdb  # Nouvel import
import requests
# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessing:
    def __init__(self):
        pass 

    def process_data(self, df: Optional[pd.DataFrame] = None):
        """
        Traite les données en ajoutant divers indicateurs techniques basés sur open_price.
        
        Args:
            df: DataFrame à traiter. Si None, utilise read_csv_to_df()
            
        Returns:
            Tuple contenant le DataFrame enrichi et sa dernière ligne.
        """
        # Si aucun DataFrame n'est fourni, lire les données
        if df is None:
            df = self.read_csv_to_df()
        
        # Vérifier que le DataFrame contient des données
        if df.empty:
            print("Le DataFrame est vide, impossible de calculer les indicateurs")
            return df, None
            
        # Vérifier que la colonne open_price existe
        if 'open_price' not in df.columns:
            print("Colonne 'open_price' manquante, impossible de calculer les indicateurs")
            return df, None
            
        # Moyennes Mobiles (période ajustée à 50 minutes pour éviter le bruit)
        df['SMA_50'] = df['open_price'].rolling(window=50).mean()
        df['EMA_50'] = df['open_price'].ewm(span=50, adjust=False).mean()
        df['min_price'] = df['open_price'].min()
        # Bollinger Bands (période ajustée à 50 minutes)
        rolling_std = df['open_price'].rolling(window=50).std()
        df['BB_middle'] = df['SMA_50']  # Utilise la SMA déjà calculée
        df['BB_upper'] = df['BB_middle'] + (rolling_std * 2)
        
        # Récupérer la dernière ligne
        last_row = df.iloc[-1].to_dict()
        
        return df, last_row
        
    def read_csv_to_df(self, file_name: str = "C:/Users/ouchaou/Desktop/ML/src/websocket/data/minute_data.csv", base_path: Optional[str] = None, min_rows: int = 50, max_wait_seconds: int = 300) -> pd.DataFrame:
        """
        Lit un fichier CSV qui se remplit continuellement. Attend jusqu'à ce que le fichier
        contienne au moins min_rows lignes avant de les extraire et les supprimer.
        
        Args:
            file_name: Nom du fichier CSV à lire (par défaut: minute_data.csv)
            base_path: Chemin de base optionnel
            min_rows: Nombre minimum de lignes à extraire (50 par défaut)
            max_wait_seconds: Temps maximum d'attente en secondes (5 minutes par défaut)
            
        Returns:
            DataFrame contenant les lignes extraites ou DataFrame vide si temps d'attente dépassé
        """
        # Déterminer le chemin du fichier
        if os.path.isabs(file_name):
            file_path = file_name
        else:
            if base_path is None:
                base_path = os.path.join("src", "websocket", "data")
            file_path = os.path.join(base_path, file_name)
            
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            print(f"Le fichier {file_path} n'existe pas")
            return pd.DataFrame()
        
        # Variables pour suivre l'attente
        start_time = time.time()
        wait_interval = 10  # Vérifier toutes les 10 secondes
        
        while True:
            try:
                # Vérifier le temps d'attente écoulé
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_seconds:
                    print(f"Temps d'attente maximum dépassé ({max_wait_seconds}s). Abandon.")
                    return pd.DataFrame()
                
                # Vérifier le nombre de lignes dans le fichier
                with open(file_path, 'r') as f:
                    # Compter le nombre de lignes total (méthode efficace)
                    line_count = sum(1 for _ in f)
                
                # Le nombre de lignes de données est le total moins l'en-tête
                data_lines = line_count - 1 if line_count > 0 else 0
                
                # Si on a assez de lignes, on les traite
                if data_lines >= min_rows:
                    print(f"Le fichier contient {data_lines} lignes, traitement des {min_rows} premières")
                    
                    # Lire tout le fichier
                    df_full = pd.read_csv(file_path)
                    
                    # Extraire les min_rows premières lignes
                    df_extract = df_full.iloc[:min_rows].copy()
                    
                    # Conserver le reste pour réécrire le fichier
                    df_remaining = df_full.iloc[min_rows:].copy()
                    
                    # Réécrire le fichier avec seulement les lignes restantes
                    df_remaining.to_csv(file_path, index=False)
                    
                    # Traitement des colonnes
                    if 'timestamp' in df_extract.columns:
                        df_extract['timestamp_datetime'] = pd.to_datetime(df_extract['timestamp'], unit='s')
                    
                    # Renommer les colonnes pour correspondre au reste du code
                    column_mapping = {
                        'open_price': 'open_price',  # Déjà correctement nommé
                        'volume_sum': 'total_volume',
                    }
                    df_extract = df_extract.rename(columns=column_mapping)
                    
                    return df_extract
                else:
                    # Pas assez de lignes, attendre et vérifier à nouveau
                    print(f"En attente: {data_lines}/{min_rows} lignes accumulées. "
                          f"Temps écoulé: {int(elapsed_time)}s/{max_wait_seconds}s")
                    time.sleep(wait_interval)
                    
            except Exception as e:
                print(f"Erreur lors du traitement du fichier CSV: {str(e)}")
                time.sleep(wait_interval)  # Attendre un peu avant de réessayer
        
        # Cette ligne ne devrait jamais être atteinte à cause de la condition de temps max
        return pd.DataFrame()

    def monitor_csv_file(self, file_path: str, batch_size: int = 50, check_interval: float = 1.0) -> Generator[pd.DataFrame, None, None]:
        """
        Surveille en continu un fichier CSV et yield un DataFrame chaque fois qu'un nouveau lot de données est disponible.
        
        Args:
            file_path: Chemin vers le fichier CSV
            batch_size: Nombre de lignes à traiter par lot (défaut: 50)
            check_interval: Intervalle de vérification en secondes (défaut: 1.0)
            
        Yields:
            DataFrame contenant le prochain lot de données
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")

        while True:
            try:
                # Lecture efficace du nombre de lignes
                line_count = sum(1 for _ in open(file_path)) - 1  # -1 pour le header
                
                if line_count >= batch_size:
                    # Lecture optimisée avec chunks
                    df = pd.read_csv(file_path)
                    batch_df = df.iloc[:batch_size].copy()
                    
                    # Réécriture efficace du fichier sans les lignes traitées
                    df.iloc[batch_size:].to_csv(file_path, index=False)
                    
                    yield batch_df
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement du fichier: {e}")
                time.sleep(check_interval)


# # Dans votre code principal
# data_processor = DataProcessing()
# csv_path = "C:/Users/ouchaou/Desktop/ML/src/websocket/data/minute_data.csv"


# try:
#     for df_batch in data_processor.monitor_csv_file(csv_path):
#         processed_df, last_row = data_processor.process_data(df_batch)
        
#         # Conversion du dictionnaire en DataFrame pour une seule ligne
#         last_row_df = pd.DataFrame([last_row])
#         response = requests.get("http://localhost:8000/predict", params={
#         "open_price": last_row['open_price'],
#         "max_price": last_row['max_price'],
#         "min_price": last_row['min_price'],
#         "EMA_50": last_row['EMA_50'],
#         "BB_upper": last_row['BB_upper'],
#         "human_time": last_row['human_time']})

        
        
#         logger.info(f"Nouveau lot de données traité: {len(processed_df)} lignes")
#         logger.info(f"Dernière ligne insérée dans DuckDB")
# except KeyboardInterrupt:
#     logger.info("Arrêt du monitoring")

# except Exception as e:
#     logger.error(f"Erreur: {e}")
