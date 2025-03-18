# Système d'Analyse et de Prédiction pour le Trading de Crypto-monnaies

Ce projet est un système complet pour la collecte, le traitement, l'analyse et la prédiction des données de trading de crypto-monnaies. Il utilise une architecture modulaire qui combine le stockage des données historiques, le traitement en temps réel, et la prédiction à l'aide de modèles de machine learning.

## 🔑 Fonctionnalités Principales

- **Chargement de données historiques** depuis MinIO vers DuckDB
- **Streaming en temps réel** des données de trading via Websocket Finnhub
- **Traitement et enrichissement des données** avec des indicateurs techniques
- **Modèles de prédiction** basés sur Gradient Boosting
- **Visualisation interactive** des données et prédictions avec Streamlit
- **API** pour l'accès aux données et aux prédictions

## Architecture du projet

![Architecture du projet](./src/img/Diagramme%20sans%20nom.drawio.png)

## 📋 Structure du Projet

```          # Notebooks Jupyter pour l'analyse exploratoire
├── src/                   # Code source
│   ├── data/              # Module de gestion des données
│   │   └── uplode_data.py # Script de chargement des données historiques
│   ├── features/          # Extraction et traitement des caractéristiques
│   │   ├── data_processing.py      # Traitement des données
│   │   └── data_processing_api.py  # API pour le traitement des données
│   ├── models/            # Modèles de machine learning
│   │   ├── model_api/     # API pour accéder aux modèles
│   │   └── train_model/   # Scripts d'entraînement des modèles
│   ├── visualization/     # Visualisation des données et résultats
│   │   └── app.py         # Application Streamlit pour la visualisation
│   └── websocket/         # Module de récupération de données en temps réel
│       └── finnhub_websocket.py # Client WebSocket pour Finnhub
├── docker-compose.yml     # Configuration Docker pour MinIO
├── requirements.txt       # Dépendances Python
└── .env                   # Configuration des variables d'environnement
```

## 🛠️ Prérequis

- Python 3.8+
- MinIO (pour le stockage des données)
- Compte Finnhub avec clé API (pour les données en temps réel)

## 📦 Installation

1. **Cloner le repository**
```bash
git clone <url-du-repository>
cd <nom-du-repository>
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Configurer l'environnement**
   - Copier `.env.example` vers `.env`
   - Remplir les variables d'environnement nécessaires

4. **Démarrer MinIO avec Docker**
```bash
docker-compose up -d
```

## 🚀 Utilisation

### Chargement des données historiques

```python
from src.data.uplode_data import init_duckdb, process_file

# Initialiser la base de données
con = init_duckdb()

# Traiter un fichier CSV
process_file(con, "s3://btc/BTCUSDT-trades-2022-01.csv")
```

### Collecte des données en temps réel

```python
from src.websocket.finnhub_websocket import FinnhubWebSocket

# Initialiser et démarrer le WebSocket
ws = FinnhubWebSocket()
ws.start()  # Se connecte et commence à collecter les données
```

### Traitement des données

```python
from src.features.data_processing import DataProcessing

# Initialiser le processeur de données
processor = DataProcessing()

# Traiter un DataFrame
df_processed = processor.process_data(df)
```

### Lancer l'application de visualisation

```bash
cd src/visualization
streamlit run app.py
```

## 🧪 Entraînement des modèles

Le projet utilise principalement des modèles Gradient Boosting (GBR) pour la prédiction. Pour entraîner un nouveau modèle:

1. Accéder au notebook d'entraînement
```bash
jupyter notebook src/models/train_model/train_gbr_model.ipynb
```

2. Suivre les instructions dans le notebook pour:
   - Charger les données
   - Préparer les caractéristiques
   - Optimiser les hyperparamètres
   - Entraîner le modèle
   - Évaluer les performances
   - Sauvegarder le modèle

## ⚙️ Architecture Technique

### Stockage des données
- **MinIO**: Stockage des fichiers CSV bruts
- **DuckDB**: Base de données analytique pour le traitement rapide

### Traitement des données
- **Pandas**: Manipulation des données
- **Scikit-learn**: Prétraitement et évaluation des modèles

### Modélisation
- **XGBoost/LightGBM**: Modèles de prédiction
- **Optuna**: Optimisation des hyperparamètres

### Visualisation
- **Streamlit**: Interface utilisateur interactive
- **Plotly**: Graphiques dynamiques

## 🔒 Configuration

Le fichier `.env` doit contenir les variables suivantes:

```
# Token Finnhub pour l'API
FINNHUB_TOKEN=votre_token_finnhub

# Configuration MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=votre_access_key
MINIO_SECRET_KEY=votre_secret_key

# Configuration DuckDB
DUCKDB_MEMORY_LIMIT=15GB
DUCKDB_THREADS=8
DUCKDB_PATH=data/trades.duckdb
DUCKDB_TEMP_DIR=data/temp
```

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou une pull request.

## 📝 Licence

Ce projet est sous licence [MIT](LICENSE). 