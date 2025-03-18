import websocket
import json
import os
import logging
import time
import datetime
import csv
from typing import Optional
from dotenv import load_dotenv
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("finnhub_websocket")

class FinnhubWebSocket:
    def __init__(self, token: Optional[str] = None):
        load_dotenv()
        self.token = token or os.environ.get("FINNHUB_TOKEN")
        if not self.token:
            raise ValueError("Token Finnhub non fourni")
        
        self.ws = None
        self.opening_price = None
        self.last_reconnect_attempt = 0
        self.reconnect_delay = 5  # secondes
        
        # Pour la sauvegarde
        self.first_trade_saved = False
        self.last_hourly_save = 0
        self.latest_trade_file = "data/latest_trade.csv"  # Changé en CSV
        
        # Pour l'agrégation par minute
        self.minute_data = {
            "volume_sum": 0,
            "max_price": 0,
            "open_price": None
        }
        self.last_minute_save = 0
        self.minute_data_file = "data/minute_data.csv"  # Uniquement CSV
    
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get("type") == "trade":
                for trade in data.get("data", []):
                    # Le premier prix reçu après connexion est considéré comme prix d'ouverture
                    if self.opening_price is None:
                        self.opening_price = trade.get("p")
                        logger.info(f"Prix d'ouverture BTC/USDT: {self.opening_price}")
                    
                    # Sauvegarder la transaction actuelle (écrase le fichier)
                    self.save_trade(trade, "latest_trade")
                    
                    # Si c'est la première transaction, on le note dans les logs
                    if not self.first_trade_saved:
                        self.first_trade_saved = True
                        logger.info(f"Première transaction sauvegardée: {trade}")
                    
                    # Mise à jour des données par minute
                    current_time = time.time()
                    
                    # Si c'est la première transaction de la minute, initialiser l'open_price
                    if self.last_minute_save == 0 or self.minute_data["open_price"] is None:
                        self.minute_data["open_price"] = trade.get("p", 0)
                    
                    # Ajouter au volume total
                    self.minute_data["volume_sum"] += trade.get("v", 0)
                    
                    # Mettre à jour le prix maximum
                    current_price = trade.get("p", 0)
                    self.minute_data["max_price"] = max(self.minute_data["max_price"], current_price)
                    
                    # Vérifier si une minute s'est écoulée
                    if current_time - self.last_minute_save >= 60:  # 60 secondes = 1 minute
                        # Sauvegarder les données de la minute
                        self.append_minute_data()
                        
                        # Réinitialiser pour la prochaine minute
                        self.last_minute_save = current_time
                        self.minute_data = {
                            "volume_sum": 0,
                            "max_price": 0,
                            "open_price": current_price
                        }
                    
                    # Log pour la sauvegarde horaire
                    if current_time - self.last_hourly_save >= 3600:  # 3600 secondes = 1 heure
                        self.last_hourly_save = current_time
                        logger.info(f"Transaction horaire mise à jour: {trade}")
                    
                    logger.debug(f"Trade: {trade}")
        except json.JSONDecodeError:
            logger.error(f"Erreur de décodage JSON: {message}")
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
    
    def save_trade(self, trade, filename_prefix):
        """Sauvegarde une transaction dans un fichier CSV en écrasant le fichier existant"""
        try:
            # Créer le dossier data s'il n'existe pas
            os.makedirs("data", exist_ok=True)
            
            # Formatage du timestamp pour le rendre plus lisible
            if "t" in trade:
                trade_time = datetime.datetime.fromtimestamp(trade["t"]/1000)
                trade_with_readable_time = dict(trade)
                trade_with_readable_time["human_time"] = trade_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                trade_with_readable_time = trade
            
            # Écrire dans un fichier CSV (écrase le fichier s'il existe)
            filepath = os.path.join("data", f"{filename_prefix}.csv")
            
            # Déterminer les champs pour le CSV
            fieldnames = list(trade_with_readable_time.keys())
            
            with open(filepath, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(trade_with_readable_time)
                
            logger.debug(f"Transaction sauvegardée dans {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la transaction: {e}")
    
    def append_minute_data(self):
        """Ajoute les données agrégées de la minute au fichier CSV existant"""
        try:
            # Créer le dossier data s'il n'existe pas
            os.makedirs("data", exist_ok=True)
            
            # Récupérer le timestamp actuel
            timestamp = time.time()
            human_time = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            # Préparer les données à ajouter
            data_entry = {
                "timestamp": timestamp,
                "human_time": human_time,
                "volume_sum": self.minute_data["volume_sum"],
                "max_price": self.minute_data["max_price"],
                "open_price": self.minute_data["open_price"]
            }
            
            # Vérifier si le fichier existe déjà
            file_exists = os.path.exists(self.minute_data_file)
            
            # Ouvrir le fichier en mode append
            with open(self.minute_data_file, 'a', newline='') as f:
                fieldnames = ["timestamp", "human_time", "volume_sum", "max_price", "open_price"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Écrire l'en-tête seulement si le fichier est nouveau
                if not file_exists:
                    writer.writeheader()
                
                # Ajouter les nouvelles données
                writer.writerow(data_entry)
            
            logger.info(f"Données de la minute ajoutées au fichier CSV: {data_entry}")
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des données de la minute: {e}")
    
    def json_to_csv(self, json_file, csv_file):
        """Convertit un fichier JSON existant en format CSV"""
        try:
            if not os.path.exists(json_file):
                logger.warning(f"Le fichier JSON {json_file} n'existe pas")
                return False
            
            # Lire les données JSON
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Vérifier si les données sont une liste
            if not isinstance(data, list):
                data = [data]  # Convertir en liste si c'est un seul objet
            
            # S'assurer que le dossier existe
            os.makedirs(os.path.dirname(csv_file) or '.', exist_ok=True)
            
            # Écrire les données en CSV
            if data:
                fieldnames = data[0].keys()
                with open(csv_file, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
                logger.info(f"Fichier JSON {json_file} converti avec succès en CSV {csv_file}")
                return True
            else:
                logger.warning(f"Aucune donnée à convertir depuis {json_file}")
                return False
        except Exception as e:
            logger.error(f"Erreur lors de la conversion JSON vers CSV: {e}")
            return False
    
    def on_error(self, ws, error):
        logger.error(f"Erreur WebSocket: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"Connexion fermée: {close_status_code} - {close_msg}")
        self._reconnect()
    
    def on_open(self, ws):
        logger.info("Connexion établie avec Finnhub")
        ws.send(json.dumps({
            "type": "subscribe",
            "symbol": "BINANCE:BTCUSDT"
        }))
    
    def _reconnect(self):
        current_time = time.time()
        if current_time - self.last_reconnect_attempt > self.reconnect_delay:
            self.last_reconnect_attempt = current_time
            logger.info(f"Tentative de reconnexion dans {self.reconnect_delay} secondes...")
            time.sleep(self.reconnect_delay)
            self.start()
    
    def start(self):
        try:
            url = f"wss://ws.finnhub.io?token={self.token}"
            self.ws = websocket.WebSocketApp(
                url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du WebSocket: {e}")
            self._reconnect()
    
    def stop(self):
        if self.ws:
            self.ws.close()

if __name__ == "__main__":
    # Définir FINNHUB_TOKEN dans les variables d'environnement ou le passer en paramètre
    client = FinnhubWebSocket()
    try:
        client.start()
    except KeyboardInterrupt:
        logger.info("Arrêt manuel du client")
        client.stop()
