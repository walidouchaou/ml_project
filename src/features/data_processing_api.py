from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, FilePath
from typing import Optional, Dict, List
import pandas as pd
import asyncio
from datetime import datetime
import logging
import os
from contextlib import asynccontextmanager
import json

# Configuration avancée du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringConfig(BaseModel):
    file_path: str
    batch_size: int = 50
    check_interval: float = 1.0
    max_retries: int = 3
    timeout: int = 300

class MonitoringStatus(BaseModel):
    is_running: bool
    file_path: str
    last_processed: Optional[datetime]
    processed_batches: int
    last_error: Optional[str]

class DataBatch(BaseModel):
    timestamp: datetime
    batch_id: int
    data: List[Dict]
    metadata: Dict

class DataState:
    def __init__(self):
        self.latest_data: Optional[Dict] = None
        self.is_monitoring: bool = False
        self.last_update: Optional[datetime] = None
        self.error: Optional[str] = None

data_state = DataState()

class MonitoringService:
    def __init__(self):
        self.active_monitors: Dict[str, bool] = {}
        self.monitoring_stats: Dict[str, MonitoringStatus] = {}
        self.batch_counter: Dict[str, int] = {}

    async def process_batch(self, df: pd.DataFrame, file_path: str) -> DataBatch:
        """Traite un lot de données et prépare la réponse"""
        batch_id = self.batch_counter.get(file_path, 0) + 1
        self.batch_counter[file_path] = batch_id

        return DataBatch(
            timestamp=datetime.now(),
            batch_id=batch_id,
            data=df.to_dict('records'),
            metadata={
                "rows_count": len(df),
                "columns": list(df.columns)
            }
        )

    async def monitor_file(self, config: MonitoringConfig) -> AsyncGenerator[DataBatch, None]:
        """Version asynchrone et optimisée du monitoring de fichier"""
        if not os.path.exists(config.file_path):
            raise FileNotFoundError(f"Le fichier {config.file_path} n'existe pas")

        self.active_monitors[config.file_path] = True
        self.monitoring_stats[config.file_path] = MonitoringStatus(
            is_running=True,
            file_path=config.file_path,
            last_processed=None,
            processed_batches=0,
            last_error=None
        )

        try:
            while self.active_monitors[config.file_path]:
                try:
                    # Lecture efficace du nombre de lignes
                    line_count = sum(1 for _ in open(config.file_path)) - 1

                    if line_count >= config.batch_size:
                        # Lecture optimisée avec chunks
                        df = pd.read_csv(config.file_path)
                        batch_df = df.iloc[:config.batch_size].copy()
                        
                        # Réécriture optimisée du fichier
                        df.iloc[config.batch_size:].to_csv(config.file_path, index=False)
                        
                        # Traitement du lot
                        batch_data = await self.process_batch(batch_df, config.file_path)
                        
                        # Mise à jour des statistiques
                        self.monitoring_stats[config.file_path].last_processed = datetime.now()
                        self.monitoring_stats[config.file_path].processed_batches += 1
                        
                        yield batch_data

                    await asyncio.sleep(config.check_interval)

                except Exception as e:
                    logger.error(f"Erreur lors du monitoring: {str(e)}")
                    self.monitoring_stats[config.file_path].last_error = str(e)
                    await asyncio.sleep(config.check_interval)

        finally:
            self.active_monitors[config.file_path] = False
            self.monitoring_stats[config.file_path].is_running = False

app = FastAPI(
    title="Data Monitoring API",
    description="API de surveillance des données en temps réel",
    version="1.0.0"
)

monitoring_service = MonitoringService()

class DataProcessor:
    def __init__(self):
        self.processor = DataProcessing()

    async def process_and_store_data(self, df: pd.DataFrame) -> None:
        """Traite les données et stocke la dernière ligne"""
        try:
            # Traitement des données
            processed_df = self.processor.process_data(df)
            
            if not processed_df.empty:
                # Conversion de la dernière ligne en dictionnaire
                latest_row = processed_df.iloc[-1].to_dict()
                
                # Ajout des métadonnées
                data_state.latest_data = {
                    "data": latest_row,
                    "timestamp": datetime.now().isoformat(),
                    "batch_size": len(processed_df)
                }
                data_state.last_update = datetime.now()
                
                logger.info(f"Nouvelles données traitées: {len(processed_df)} lignes")
        
        except Exception as e:
            data_state.error = str(e)
            logger.error(f"Erreur lors du traitement: {e}")

async def monitor_data(config: MonitoringConfig):
    """Fonction de monitoring asynchrone"""
    data_processor = DataProcessor()
    data_state.is_monitoring = True
    
    try:
        async def process_batch(batch_df: pd.DataFrame):
            await data_processor.process_and_store_data(batch_df)

        while data_state.is_monitoring:
            try:
                if not os.path.exists(config.file_path):
                    raise FileNotFoundError(f"Fichier non trouvé: {config.file_path}")

                # Lecture du fichier
                line_count = sum(1 for _ in open(config.file_path)) - 1

                if line_count >= config.batch_size:
                    df = pd.read_csv(config.file_path)
                    batch_df = df.iloc[:config.batch_size].copy()
                    
                    # Réécriture du fichier sans les lignes traitées
                    df.iloc[config.batch_size:].to_csv(config.file_path, index=False)
                    
                    # Traitement asynchrone des données
                    await process_batch(batch_df)

                await asyncio.sleep(config.check_interval)

            except Exception as e:
                data_state.error = str(e)
                logger.error(f"Erreur dans la boucle de monitoring: {e}")
                await asyncio.sleep(config.check_interval)

    except Exception as e:
        data_state.error = str(e)
        logger.error(f"Erreur fatale dans le monitoring: {e}")
    finally:
        data_state.is_monitoring = False

@app.post("/monitor/start")
async def start_monitoring(config: MonitoringConfig, background_tasks: BackgroundTasks):
    """Démarre le monitoring du fichier"""
    if data_state.is_monitoring:
        raise HTTPException(
            status_code=400,
            detail="Le monitoring est déjà en cours"
        )

    background_tasks.add_task(monitor_data, config)
    
    return {
        "status": "success",
        "message": "Monitoring démarré",
        "config": config.dict()
    }

@app.get("/monitor/latest")
async def get_latest_data():
    """Récupère la dernière ligne de données traitée"""
    if not data_state.is_monitoring:
        raise HTTPException(
            status_code=400,
            detail="Le monitoring n'est pas actif"
        )

    if data_state.error:
        return {
            "status": "error",
            "error": data_state.error,
            "last_update": data_state.last_update
        }

    if data_state.latest_data is None:
        return {
            "status": "waiting",
            "message": "Aucune donnée disponible pour le moment"
        }

    return {
        "status": "success",
        "data": data_state.latest_data,
        "last_update": data_state.last_update
    }

@app.post("/monitor/stop")
async def stop_monitoring():
    """Arrête le monitoring"""
    if not data_state.is_monitoring:
        raise HTTPException(
            status_code=400,
            detail="Le monitoring n'est pas actif"
        )

    data_state.is_monitoring = False
    return {"status": "success", "message": "Monitoring arrêté"}

@app.get("/monitor/status")
async def get_status():
    """Récupère le statut actuel du monitoring"""
    return {
        "is_monitoring": data_state.is_monitoring,
        "last_update": data_state.last_update,
        "error": data_state.error
    }

@app.get("/monitor/list")
async def list_monitored_files():
    """Liste tous les fichiers en cours de monitoring"""
    return {
        "active_monitors": [
            {
                "file_path": file_path,
                "status": monitoring_service.monitoring_stats[file_path]
            }
            for file_path in monitoring_service.active_monitors
            if monitoring_service.active_monitors[file_path]
        ]
    } 