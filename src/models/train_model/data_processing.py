import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self._bdd_path = "C:/Users/ouchaou/Desktop/Project_data_portfolio/data_pipeline_streaming_finnhub/src/duckdb/data/trades.duckdb"

    def load_data(self):
        conn = duckdb.connect(self._bdd_path)
        query = "SELECT * FROM trades_aggregated_min"
        df = conn.execute(query).fetch_df()
        return df
    
    def add_time_features(self, df):
        """
        Ajoute des caractéristiques temporelles au DataFrame.
        Suppose que l'index du DataFrame est un DateTimeIndex.
        """
        # Vérifier que l'index est bien un DateTimeIndex
        if not hasattr(df.index, 'hour'):
            df = df.set_index('timestamp') if 'timestamp' in df.columns else df
            
        # Ajouter les caractéristiques temporelles
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        return df
    
    def add_time_features(self):
        """
        Ajoute des caractéristiques temporelles directement dans DuckDB.
        Retourne un DataFrame avec les caractéristiques temporelles.
        """
        conn = duckdb.connect(self._bdd_path)
        
        # Créer une vue temporaire avec les caractéristiques temporelles
        query = """
        create or replace view trades_aggregated_min_with_features as
        SELECT 
            *,
            EXTRACT(HOUR FROM CAST(datetime AS TIMESTAMP)) AS hour,
            EXTRACT(DOW FROM CAST(datetime AS TIMESTAMP)) AS day_of_week,
            EXTRACT(MONTH FROM CAST(datetime AS TIMESTAMP)) AS month
        FROM trades_aggregated_min
        """
        
        # Exécuter la requête et récupérer les résultats
        conn.execute(query)
        conn.close()
        
    
    def add_price_lags_duckdb(self, num_lags=5):
        """
        Ajoute des variables de retard (lags) pour le prix de clôture directement dans DuckDB
        en modifiant la vue trades_aggregated_min_with_features.
        
        Args:
            num_lags (int): Nombre de lags à créer (par défaut: 5)
        """
        # conn = duckdb.connect(self._bdd_path)
        
        # Construire dynamiquement la partie de la requête pour les lags
        lag_columns = []
        for lag in range(1, num_lags + 1):
            lag_columns.append(f"LAG(close_price, {lag}) OVER (ORDER BY CAST(datetime AS TIMESTAMP)) AS close_price_lag_{lag}")
        
        lag_sql = ", ".join(lag_columns)
        
        # Requête pour créer ou remplacer la vue avec les caractéristiques temporelles et les lags
        query = f"""
        CREATE OR REPLACE VIEW trades_aggregated_min_with_features_final AS
        SELECT 
            *,
            EXTRACT(HOUR FROM CAST(datetime AS TIMESTAMP)) AS hour,
            EXTRACT(DOW FROM CAST(datetime AS TIMESTAMP)) AS day_of_week,
            EXTRACT(MONTH FROM CAST(datetime AS TIMESTAMP)) AS month,
            {lag_sql}
        FROM trades_aggregated_min
        """
        
        # Exécuter la requête
        return query

    def add_technical_indicators(self):
        """
        Récupère les données depuis DuckDB et ajoute des indicateurs techniques
        adaptés pour des données à la minute avec des périodes ajustées.
        
        Returns:
            DataFrame: DataFrame avec les indicateurs techniques ajoutés
        """
        # Connexion à DuckDB et récupération des données
        conn = duckdb.connect(self._bdd_path)
        query = "SELECT * FROM trades_aggregated_min_with_features_final"
        df = conn.execute(query).fetch_df()
        conn.close()
        
        # Assurez-vous que datetime est correctement formaté
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Moyennes Mobiles (période ajustée à 50 minutes pour éviter le bruit)
        # SMA: Moyenne mobile simple - identifie la tendance générale en donnant un poids égal à toutes les observations
        # Utile pour déterminer si le prix est au-dessus/en-dessous de la tendance moyenne
        df['SMA_50'] = df['close_price'].rolling(window=50).mean()
        
        # EMA: Moyenne mobile exponentielle - donne plus de poids aux observations récentes
        # Réagit plus rapidement aux changements de prix récents que la SMA
        df['EMA_50'] = df['close_price'].ewm(span=50, adjust=False).mean()
        
        # Bollinger Bands (période ajustée à 50 minutes)
        # Mesure la volatilité relative autour de la moyenne mobile
        # Utilisé pour identifier les conditions de surachat/survente et la compression/expansion de volatilité
        rolling_std = df['close_price'].rolling(window=50).std()
        df['BB_middle'] = df['SMA_50']  # Utilise la SMA déjà calculée
        df['BB_upper'] = df['BB_middle'] + (rolling_std * 2)  # Bande supérieure = SMA + 2*écart-type
        df['BB_lower'] = df['BB_middle'] - (rolling_std * 2)  # Bande inférieure = SMA - 2*écart-type
        
        # RSI (période ajustée à 14 * 5 minutes = 70 minutes)
        # Oscillateur qui mesure la vitesse et le changement des mouvements de prix
        # RSI > 70 indique généralement une condition de surachat, RSI < 30 indique une condition de survente
        rsi_period = 14 * 5
        delta = df['close_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # Calcul du RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (période ajustée pour des cycles horaires)
        # Indicateur de tendance qui montre la relation entre deux moyennes mobiles
        # Utile pour identifier les changements de force, direction, momentum et durée d'une tendance
        fast_period = 12 * 5  # Période rapide ajustée pour données à la minute
        slow_period = 26 * 5  # Période lente ajustée pour données à la minute
        signal_period = 9 * 5  # Période du signal ajustée pour données à la minute
        
        exp1 = df['close_price'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close_price'].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = exp1 - exp2  # La ligne MACD est la différence entre les deux EMAs
        df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()  # La ligne de signal est une EMA du MACD
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']  # L'histogramme représente la différence entre MACD et signal
        
        # Volatilité (périodes ajustées à 7h et 30h)
        # Mesure l'ampleur des mouvements de prix sur différentes périodes
        # Utile pour quantifier le risque et identifier les périodes de forte/faible volatilité
        df['volatility_7h'] = df['close_price'].rolling(window=7*60).std()  # Volatilité sur 7 heures
        df['volatility_30h'] = df['close_price'].rolling(window=30*60).std()  # Volatilité sur 30 heures
        
        return df

    def store_technical_indicators(self, table_name="trades_with_indicators"):
        """
        Calcule les indicateurs techniques et les stocke dans une nouvelle table DuckDB.
        
        Args:
            table_name (str): Nom de la table où stocker les données avec indicateurs
        
        Returns:
            bool: True si l'opération a réussi
        """
        # Calculer les indicateurs techniques
        query = self.add_price_lags_duckdb()
        conn = duckdb.connect(self._bdd_path)
        conn.execute(query)
        conn.close()
        df_with_indicators = self.add_technical_indicators()
        
        
        # Remplacer les valeurs NaN par NULL pour DuckDB
        df_with_indicators = df_with_indicators.replace({np.nan: None})
        
        # Connexion à DuckDB
        conn = duckdb.connect(self._bdd_path)
        
        try:
            # Créer ou remplacer la table
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Créer la table et insérer les données
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_with_indicators")
            
            # Créer également une vue pour faciliter l'accès
            conn.execute(f"""
            CREATE OR REPLACE VIEW {table_name}_view AS
            SELECT * FROM {table_name}
            """)
            
            conn.close()
            return True
        
        except Exception as e:
            print(f"Erreur lors du stockage des indicateurs techniques: {e}")
            conn.close()
            return False

    def prepare_data_for_ml(self):
        """
        Prépare les données pour l'apprentissage automatique en:
        1. Calculant les indicateurs techniques
        2. Supprimant les valeurs manquantes
        3. Normalisant les caractéristiques numériques
        
        Returns:
            DataFrame: DataFrame prêt pour l'apprentissage automatique
        """
        # Récupérer les données avec les indicateurs techniques
        df = self.add_technical_indicators()
        
        # Supprimer les valeurs manquantes générées par les transformations
        df.dropna(inplace=True)
        
        # Normalisation des données numériques
        scaler = StandardScaler()
        features_to_scale = [
            'close_price', 'SMA_50', 'EMA_50', 
            'BB_upper', 'BB_middle', 'BB_lower', 
            'RSI', 'MACD', 'MACD_signal', 
            'volatility_7h', 'volatility_30h'
        ]
        
        # Appliquer la normalisation
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        
        # Sauvegarder le scaler pour une utilisation future (prédictions)
        self.scaler = scaler
        
        return df

    def store_ml_ready_data(self, table_name="ml_ready_data"):
        """
        Prépare les données pour le ML (indicateurs techniques, suppression des NaN, 
        normalisation) et les stocke dans une table DuckDB.
        
        Args:
            table_name (str): Nom de la table où stocker les données préparées
            
        Returns:
            bool: True si l'opération a réussi
        """
        # Préparer les données
        df = self.prepare_data_for_ml()
        
        # Connexion à DuckDB
        conn = duckdb.connect(self._bdd_path)
        
        try:
            # Créer ou remplacer la table
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Créer la table et insérer les données
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            # Sauvegarder le scaler dans un fichier pour une utilisation future
            scaler_dir = os.path.dirname(self._bdd_path)
            scaler_path = os.path.join(scaler_dir, "standard_scaler.pkl")
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"Données ML prêtes stockées dans la table '{table_name}'")
            print(f"Scaler sauvegardé dans '{scaler_path}'")
            
            conn.close()
            return True
        
        except Exception as e:
            print(f"Erreur lors du stockage des données ML: {e}")
            conn.close()
            return False



    

data_processor = DataProcessor()
data_processor.store_ml_ready_data()


