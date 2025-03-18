import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import STL
from scipy import signal
import pywt 
import matplotlib.dates as mdates
from arch import arch_model
import statsmodels.api as sm
from sklearn.cluster import KMeans

class BTCPredictor:
    """Classe pour la prédiction des prix du BTCUSD avec plusieurs modèles"""
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path("../duckdb/data/trades.duckdb").resolve()
        try:
            self.con = duckdb.connect(str(data_path))
        except Exception as e:
            raise ConnectionError(f"Impossible de se connecter à la base de données: {e}")
        
        # Attributs pour stocker les données
        self.data = None
        self.data_daily = None
        self.data_weekly = None
        self.data_monthly = None

    def load_data(self, start_date="2021-01-01", end_date="2025-03-06", limit=None):
        """Charge les données nécessaires pour la prédiction"""
        query = f"""
            SELECT 
                max_price, min_price, open_price, close_price, 
                total_volume, quote_volume, trade_count, price_range, 
                volatility_pct, datetime
            FROM trades_aggregated_min
            WHERE datetime BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY datetime
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        self.data = self.con.execute(query).fetch_df()
        # Conversion du datetime en format datetime
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        # Création d'un index temporel
        self.data.set_index('datetime', inplace=True)
        
        return self.data
    
    def aggregate_data(self):
        """Agrège les données à différentes échelles temporelles"""
        if self.data is None:
            raise ValueError("Les données n'ont pas été chargées. Appelez load_data() d'abord.")
        
        # Agrégation journalière
        self.data_daily = self.data.resample('D').agg({
            'open_price': 'first',
            'close_price': 'last',
            'max_price': 'max',
            'min_price': 'min',
            'total_volume': 'sum',
            'quote_volume': 'sum',
            'trade_count': 'sum',
            'price_range': 'mean',
            'volatility_pct': 'mean'
        })
        
        # Rendements journaliers
        self.data_daily['daily_return'] = self.data_daily['close_price'].pct_change() * 100
        self.data_daily['log_return'] = np.log(self.data_daily['close_price'] / self.data_daily['close_price'].shift(1))
        
        # Agrégation hebdomadaire
        self.data_weekly = self.data.resample('W').agg({
            'open_price': 'first',
            'close_price': 'last',
            'max_price': 'max',
            'min_price': 'min',
            'total_volume': 'sum',
            'quote_volume': 'sum',
            'trade_count': 'sum',
            'price_range': 'mean',
            'volatility_pct': 'mean'
        })
        
        # Agrégation mensuelle
        self.data_monthly = self.data.resample('M').agg({
            'open_price': 'first',
            'close_price': 'last',
            'max_price': 'max',
            'min_price': 'min',
            'total_volume': 'sum',
            'quote_volume': 'sum',
            'trade_count': 'sum',
            'price_range': 'mean',
            'volatility_pct': 'mean'
        })
        
        return {
            'daily': self.data_daily,
            'weekly': self.data_weekly,
            'monthly': self.data_monthly
        }
    
    def analyse_distribution_evolution(self, save_plots=False, output_dir='plots'):
        """Examine la distribution des prix et leur évolution temporelle"""
        if self.data_daily is None:
            self.aggregate_data()
        
        # Créer le répertoire pour les plots si nécessaire
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Évolution du prix de clôture à différentes échelles
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
        
        # Prix journaliers
        axes[0].plot(self.data_daily.index, self.data_daily['close_price'], label='Prix journalier', color='blue')
        axes[0].set_title('Évolution du prix du Bitcoin (Journalier)')
        axes[0].set_ylabel('Prix en USDT')
        axes[0].grid(True)
        axes[0].legend()
        
        # Prix hebdomadaires
        axes[1].plot(self.data_weekly.index, self.data_weekly['close_price'], label='Prix hebdomadaire', color='green')
        axes[1].set_title('Évolution du prix du Bitcoin (Hebdomadaire)')
        axes[1].set_ylabel('Prix en USDT')
        axes[1].grid(True)
        axes[1].legend()
        
        # Prix mensuels
        axes[2].plot(self.data_monthly.index, self.data_monthly['close_price'], label='Prix mensuel', color='red')
        axes[2].set_title('Évolution du prix du Bitcoin (Mensuel)')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Prix en USDT')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'evolution_prix_multi_echelles.png', dpi=300)
        plt.show()
        
        # 2. Analyse des rendements journaliers
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Histogramme des rendements journaliers
        sns.histplot(self.data_daily['daily_return'].dropna(), kde=True, ax=axes[0, 0], bins=50)
        axes[0, 0].set_title('Distribution des rendements journaliers')
        axes[0, 0].set_xlabel('Rendement (%)')
        
        # Q-Q plot des rendements journaliers
        stats.probplot(self.data_daily['daily_return'].dropna(), plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot des rendements journaliers')
        
        # Histogramme des rendements logarithmiques
        sns.histplot(self.data_daily['log_return'].dropna(), kde=True, ax=axes[1, 0], bins=50)
        axes[1, 0].set_title('Distribution des rendements logarithmiques')
        axes[1, 0].set_xlabel('Log-rendement')
        
        # Q-Q plot des rendements logarithmiques
        stats.probplot(self.data_daily['log_return'].dropna(), plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot des rendements logarithmiques')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'distribution_rendements.png', dpi=300)
        plt.show()
        
        # 3. Évolution des rendements journaliers
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Prix
        axes[0].plot(self.data_daily.index, self.data_daily['close_price'])
        axes[0].set_title('Prix du Bitcoin')
        axes[0].set_ylabel('Prix en USDT')
        axes[0].grid(True)
        
        # Rendements
        axes[1].plot(self.data_daily.index, self.data_daily['daily_return'], color='green')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_title('Rendements journaliers')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Rendement (%)')
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'evolution_rendements.png', dpi=300)
        plt.show()
        
        # 4. Comparaison de la volatilité sur différentes périodes
        rolling_windows = [7, 30, 90]
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for window in rolling_windows:
            # Calcul de la volatilité annualisée sur fenêtre glissante
            volatility = self.data_daily['daily_return'].rolling(window=window).std() * np.sqrt(365)
            ax.plot(self.data_daily.index, volatility, label=f'Volatilité {window} jours')
        
        ax.set_title('Volatilité du Bitcoin sur différentes fenêtres temporelles')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatilité annualisée (%)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'volatilite_multi_periodes.png', dpi=300)
        plt.show()
        
        # 5. Statistiques descriptives des rendements
        returns_stats = self.data_daily['daily_return'].dropna().describe()
        
        # Ajout de statistiques supplémentaires
        returns_stats['skewness'] = stats.skew(self.data_daily['daily_return'].dropna())
        returns_stats['kurtosis'] = stats.kurtosis(self.data_daily['daily_return'].dropna())
        
        # Test de normalité
        shapiro_test = stats.shapiro(self.data_daily['daily_return'].dropna())
        jarque_bera_test = stats.jarque_bera(self.data_daily['daily_return'].dropna())
        
        print("Statistiques des rendements journaliers:")
        print(returns_stats)
        print(f"\nTest de Shapiro-Wilk pour la normalité: W={shapiro_test[0]:.4f}, p-value={shapiro_test[1]:.9f}")
        print(f"Test de Jarque-Bera pour la normalité: statistique={jarque_bera_test[0]:.4f}, p-value={jarque_bera_test[1]:.9f}")
        
        # 6. Analyse par échelle logarithmique
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.semilogy(self.data_daily.index, self.data_daily['close_price'])
        ax.set_title('Évolution du prix du Bitcoin (échelle logarithmique)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix en USDT (échelle log)')
        ax.grid(True, which="both", ls="--")
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'prix_echelle_log.png', dpi=300)
        plt.show()
        
        # 7. Détection des régimes de marché
        # Calcul de la moyenne mobile 50 jours
        self.data_daily['sma_50'] = self.data_daily['close_price'].rolling(window=50).mean()
        # Calcul de la moyenne mobile 200 jours
        self.data_daily['sma_200'] = self.data_daily['close_price'].rolling(window=200).mean()
        
        # Création d'une colonne pour le régime de marché
        self.data_daily['market_regime'] = 'Neutre'
        self.data_daily.loc[self.data_daily['sma_50'] > self.data_daily['sma_200'], 'market_regime'] = 'Haussier'
        self.data_daily.loc[self.data_daily['sma_50'] < self.data_daily['sma_200'], 'market_regime'] = 'Baissier'
        
        # Visualisation des régimes de marché
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(self.data_daily.index, self.data_daily['close_price'], label='Prix', alpha=0.7)
        ax.plot(self.data_daily.index, self.data_daily['sma_50'], label='SMA 50j', linestyle='--')
        ax.plot(self.data_daily.index, self.data_daily['sma_200'], label='SMA 200j', linestyle='-.')
        
        # Coloration de l'arrière-plan selon le régime
        bullish = self.data_daily[self.data_daily['market_regime'] == 'Haussier']
        bearish = self.data_daily[self.data_daily['market_regime'] == 'Baissier']
        
        for i in range(len(bullish) - 1):
            if i < len(bullish) - 1 and bullish.index[i+1] == bullish.index[i] + timedelta(days=1):
                ax.axvspan(bullish.index[i], bullish.index[i+1], alpha=0.2, color='green')
        
        for i in range(len(bearish) - 1):
            if i < len(bearish) - 1 and bearish.index[i+1] == bearish.index[i] + timedelta(days=1):
                ax.axvspan(bearish.index[i], bearish.index[i+1], alpha=0.2, color='red')
        
        ax.set_title('Régimes de marché du Bitcoin (Haussier vs Baissier)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix en USDT')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'regimes_marche.png', dpi=300)
        plt.show()
        
        # 8. Test de stationnarité (Augmented Dickey-Fuller)
        from statsmodels.tsa.stattools import adfuller
        
        # Test ADF sur le prix
        result_price = adfuller(self.data_daily['close_price'].dropna())
        # Test ADF sur les rendements
        result_returns = adfuller(self.data_daily['daily_return'].dropna())
        
        print("\nTest de stationnarité - Augmented Dickey-Fuller:")
        print(f"Prix - Statistique ADF: {result_price[0]:.4f}, p-value: {result_price[1]:.9f}")
        print(f"Rendements - Statistique ADF: {result_returns[0]:.4f}, p-value: {result_returns[1]:.9f}")
        print("Interprétation: Une p-value < 0.05 indique que nous pouvons rejeter l'hypothèse nulle de non-stationnarité")
        
        return {
            'returns_stats': returns_stats,
            'shapiro_test': shapiro_test,
            'jarque_bera_test': jarque_bera_test,
            'adf_price': result_price,
            'adf_returns': result_returns
        }

    def analyse_tendances_saisonnieres(self, save_plots=False, output_dir='plots'):
        """Analyse les tendances saisonnières et les cycles dans les données Bitcoin"""
        if self.data_daily is None:
            self.aggregate_data()
            
        # Créer le répertoire pour les plots si nécessaire
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
        # 1. Décomposition STL (Seasonal-Trend decomposition using LOESS)
        print("Décomposition STL des prix du Bitcoin...")
        # Utiliser une période de 365 jours pour capturer les cycles annuels
        clean_price_data = self.data_daily['close_price'].dropna()
        stl = STL(clean_price_data, period=365)
        result = stl.fit()

        # Extraire les composantes
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid

        # Visualiser la décomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        axes[0].plot(clean_price_data.index, clean_price_data)
        axes[0].set_title('Prix original')
        axes[0].set_ylabel('Prix en USDT')
        axes[0].grid(True)

        axes[1].plot(clean_price_data.index, trend)
        axes[1].set_title('Tendance')
        axes[1].set_ylabel('Prix en USDT')
        axes[1].grid(True)

        axes[2].plot(clean_price_data.index, seasonal)
        axes[2].set_title('Saisonnalité')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True)

        axes[3].plot(clean_price_data.index, residual)
        axes[3].set_title('Résidus')
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('Amplitude')
        axes[3].grid(True)

        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'decomposition_stl.png', dpi=300)
        plt.show()
        
        # 2. Analyse spectrale pour identifier les fréquences dominantes
        print("Analyse spectrale des rendements du Bitcoin...")
        returns = self.data_daily['daily_return'].dropna().values
        
        # Calculer le périodogramme
        f, Pxx = signal.periodogram(returns)
        
        # Convertir les fréquences en périodes (jours)
        periods = 1/f[1:]  # Ignorer la fréquence zéro
        power = Pxx[1:]
        
        # Visualiser le spectre de puissance
        plt.figure(figsize=(12, 6))
        plt.plot(periods, power)
        plt.xscale('log')
        plt.xlabel('Période (jours)')
        plt.ylabel('Puissance spectrale')
        plt.title('Analyse spectrale des rendements du Bitcoin')
        plt.grid(True)
        
        # Identifier les pics (périodes dominantes)
        peak_indices = signal.find_peaks(power)[0]
        dominant_periods = periods[peak_indices]
        dominant_powers = power[peak_indices]
        
        # Trier par puissance et prendre les 5 plus importantes
        sorted_indices = np.argsort(dominant_powers)[::-1][:5]
        top_periods = dominant_periods[sorted_indices]
        
        print(f"Périodes dominantes détectées: {', '.join([f'{p:.1f} jours' for p in top_periods])}")
        
        # Marquer les périodes dominantes sur le graphique
        for period, pow_val in zip(dominant_periods[sorted_indices], dominant_powers[sorted_indices]):
            plt.annotate(f"{period:.1f}j", 
                         xy=(period, pow_val),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center')
            plt.plot([period], [pow_val], 'ro')
            
        if save_plots:
            plt.savefig(output_path / 'analyse_spectrale.png', dpi=300)
        plt.show()
        
        # 3. Heatmaps temporelles pour visualiser les effets saisonniers
        print("Création de heatmaps des rendements mensuels...")
        
        # Créer un DataFrame avec année et mois comme indices
        returns_by_month = pd.DataFrame({
            'year': self.data_daily.index.year,
            'month': self.data_daily.index.month,
            'return': self.data_daily['daily_return']
        }).dropna()
        
        # Calculer le rendement moyen par mois et année
        monthly_returns = returns_by_month.groupby(['year', 'month'])['return'].mean().unstack()
        
        # Créer la heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
        plt.title('Rendements mensuels moyens du Bitcoin (%)')
        plt.xlabel('Mois')
        plt.ylabel('Année')
        
        if save_plots:
            plt.savefig(output_path / 'heatmap_rendements_mensuels.png', dpi=300)
        plt.show()
        
        # 4. Effets de calendrier et cycles hebdomadaires
        print("Analyse des effets de calendrier...")
        
        # Ajouter des colonnes pour le jour de la semaine et le mois
        self.data_daily['day_of_week'] = self.data_daily.index.dayofweek
        self.data_daily['month'] = self.data_daily.index.month
        self.data_daily['is_weekend'] = self.data_daily['day_of_week'].isin([5, 6]).astype(int)
        self.data_daily['is_month_end'] = self.data_daily.index.is_month_end.astype(int)
        self.data_daily['is_quarter_end'] = self.data_daily.index.is_quarter_end.astype(int)
        
        # Analyser les rendements par jour de la semaine
        day_returns = self.data_daily.groupby('day_of_week')['daily_return'].agg(['mean', 'std'])
        day_returns.index = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        
        # Visualiser les rendements par jour de la semaine
        plt.figure(figsize=(12, 6))
        day_returns['mean'].plot(kind='bar', yerr=day_returns['std'], capsize=5)
        plt.title('Rendements moyens par jour de la semaine')
        plt.ylabel('Rendement moyen (%)')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True, axis='y')
        
        if save_plots:
            plt.savefig(output_path / 'rendements_jour_semaine.png', dpi=300)
        plt.show()
        
        # Analyser les rendements par mois
        month_returns = self.data_daily.groupby('month')['daily_return'].agg(['mean', 'std'])
        month_returns.index = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        
        # Visualiser les rendements par mois
        plt.figure(figsize=(12, 6))
        month_returns['mean'].plot(kind='bar', yerr=month_returns['std'], capsize=5)
        plt.title('Rendements moyens par mois')
        plt.ylabel('Rendement moyen (%)')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True, axis='y')
        
        if save_plots:
            plt.savefig(output_path / 'rendements_mensuels.png', dpi=300)
        plt.show()
        
        # 5. Corrélation avec les événements majeurs du marché crypto
        print("Analyse de l'impact des halvings Bitcoin...")
        
        # Définir les dates des halvings Bitcoin
        halving_dates = ['2020-05-11', '2024-04-20']  # Dates approximatives
        
        # Créer une fonction pour analyser les rendements autour des événements
        def analyze_event_impact(event_dates, window=30):
            event_returns = pd.DataFrame()
            
            for date in event_dates:
                event_date = pd.to_datetime(date)
                start_date = event_date - pd.Timedelta(days=window)
                end_date = event_date + pd.Timedelta(days=window)
                
                # Vérifier si l'événement est dans notre plage de données
                if start_date < self.data_daily.index.min() or end_date > self.data_daily.index.max():
                    print(f"Avertissement: L'événement du {date} est partiellement ou totalement hors de la plage de données")
                    continue
                
                # Extraire les rendements autour de l'événement
                event_window = self.data_daily.loc[start_date:end_date, 'daily_return']
                
                # Réindexer pour centrer sur le jour de l'événement (jour 0)
                event_window.index = (event_window.index - event_date).days
                
                # Ajouter à notre DataFrame d'événements
                event_returns[date] = event_window
            
            if event_returns.empty:
                print("Aucun événement dans la plage de données")
                return None
                
            # Calculer la moyenne des rendements pour tous les événements
            event_returns['mean'] = event_returns.mean(axis=1)
            
            # Calculer les rendements cumulés
            event_returns['cumulative'] = (1 + event_returns['mean']/100).cumprod() - 1
            
            return event_returns
        
        # Analyser l'impact des halvings
        halving_impact = analyze_event_impact(halving_dates)
        
        if halving_impact is not None:
            # Visualiser l'impact cumulé
            plt.figure(figsize=(12, 6))
            halving_impact['cumulative'].mul(100).plot()
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Impact cumulé des halvings sur le prix du Bitcoin')
            plt.xlabel('Jours relatifs au halving (0 = jour du halving)')
            plt.ylabel('Rendement cumulé (%)')
            plt.grid(True)
            
            if save_plots:
                plt.savefig(output_path / 'impact_halvings.png', dpi=300)
            plt.show()
        
        # 6. Analyse par ondelettes
        print("Analyse par ondelettes des rendements...")
        
        # Préparer les données
        returns = self.data_daily['daily_return'].dropna().values
        dates = self.data_daily.index[~np.isnan(self.data_daily['daily_return'])]
        
        # Appliquer la transformée en ondelettes continues
        scales = np.arange(1, min(128, len(returns)//2))
        coef, freqs = pywt.cwt(returns, scales, 'morl')
        
        # Visualiser le scalogramme
        plt.figure(figsize=(12, 8))
        plt.imshow(abs(coef), extent=[0, len(returns), 1, scales[-1]], aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.ylabel('Échelle (période en jours)')
        plt.xlabel('Temps')
        plt.title('Analyse par ondelettes des rendements du Bitcoin')
        
        # Ajouter des étiquettes d'échelle plus significatives
        scale_ticks = np.array([1, 7, 30, 90, 180, 365])
        scale_ticks = scale_ticks[scale_ticks < scales[-1]]
        plt.yticks(scale_ticks)
        
        # Ajouter des dates sur l'axe x
        num_dates = 6  # Nombre de dates à afficher
        date_indices = np.linspace(0, len(returns)-1, num_dates, dtype=int)
        date_labels = [dates[i].strftime('%Y-%m') for i in date_indices]
        plt.xticks(date_indices, date_labels)
        
        if save_plots:
            plt.savefig(output_path / 'analyse_ondelettes.png', dpi=300)
        plt.show()
        
        # 7. Analyse de la volatilité par jour de la semaine
        volatility_by_day = self.data_daily.groupby('day_of_week')['daily_return'].std() * np.sqrt(365)
        volatility_by_day.index = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        
        plt.figure(figsize=(12, 6))
        volatility_by_day.plot(kind='bar')
        plt.title('Volatilité annualisée par jour de la semaine')
        plt.ylabel('Volatilité (%)')
        plt.grid(True, axis='y')
        
        if save_plots:
            plt.savefig(output_path / 'volatilite_jour_semaine.png', dpi=300)
        plt.show()
        
        # Retourner les résultats des analyses
        return {
            'stl_decomposition': {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual
            },
            'dominant_periods': top_periods,
            'day_returns': day_returns,
            'month_returns': month_returns,
            'halving_impact': halving_impact
        }

    def analyse_correlations_avancees(self, save_plots=False, output_dir='plots'):
        """Analyse les corrélations entre les différentes variables du marché Bitcoin"""
        if self.data_daily is None:
            self.aggregate_data()
        
        # Créer le répertoire pour les plots si nécessaire
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        print("Analyse des corrélations entre variables...")
        
        # Préparation des données
        # Calculer le spread journalier
        self.data_daily['spread'] = self.data_daily['max_price'] - self.data_daily['min_price']
        self.data_daily['spread_pct'] = self.data_daily['spread'] / self.data_daily['close_price'] * 100
        
        # Calculer la volatilité sur 7 jours
        self.data_daily['volatility_7d'] = self.data_daily['daily_return'].rolling(window=7).std() * np.sqrt(365)
        
        # Sélectionner les variables d'intérêt
        variables = ['close_price', 'total_volume', 'quote_volume', 'trade_count', 
                     'price_range', 'volatility_pct', 'daily_return', 'spread', 'spread_pct', 'volatility_7d']
        
        # Créer un DataFrame sans valeurs manquantes
        df_corr = self.data_daily[variables].dropna()
        
        # 1. Matrice de corrélation de base
        print("Calcul de la matrice de corrélation...")
        corr_matrix = df_corr.corr()
        
        # Visualiser la matrice de corrélation
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title('Matrice de corrélation entre les variables du marché Bitcoin')
        
        if save_plots:
            plt.savefig(output_path / 'matrice_correlation.png', dpi=300)
        plt.show()
        
        # 2. Corrélations croisées décalées
        print("Analyse des corrélations croisées décalées...")
        
        # Fonction pour calculer les corrélations croisées
        def cross_correlation(x, y, max_lag=30):
            corrs = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
                else:
                    corr = np.corrcoef(x, y)[0, 1]
                corrs.append(corr)
            return np.array(corrs)
        
        # Paires de variables à analyser
        pairs = [
            ('total_volume', 'volatility_pct'),
            ('price_range', 'trade_count'),
            ('spread_pct', 'total_volume'),
            ('daily_return', 'total_volume')
        ]
        
        # Calculer et visualiser les corrélations croisées
        fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 5 * len(pairs)))
        
        for i, (var1, var2) in enumerate(pairs):
            lags = np.arange(-30, 31)
            ccf = cross_correlation(df_corr[var1].values, df_corr[var2].values)
            
            axes[i].plot(lags, ccf)
            axes[i].axhline(y=0, color='r', linestyle='-')
            axes[i].axhline(y=1.96/np.sqrt(len(df_corr)), color='r', linestyle='--')
            axes[i].axhline(y=-1.96/np.sqrt(len(df_corr)), color='r', linestyle='--')
            axes[i].set_title(f'Corrélation croisée: {var1} vs {var2}')
            axes[i].set_xlabel('Décalage (jours)')
            axes[i].set_ylabel('Corrélation')
            axes[i].grid(True)
            
            # Identifier le décalage avec la corrélation maximale
            max_corr_idx = np.argmax(np.abs(ccf))
            max_lag = lags[max_corr_idx]
            max_corr = ccf[max_corr_idx]
            
            axes[i].annotate(f'Corrélation max: {max_corr:.3f} au décalage {max_lag}',
                             xy=(max_lag, max_corr),
                             xytext=(10, 30 if max_corr > 0 else -30),
                             textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_path / 'correlations_croisees.png', dpi=300)
        plt.show()
        
        # 3. Analyse de causalité de Granger
        print("Test de causalité de Granger...")
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Paires à tester pour la causalité
        granger_pairs = [
            ('total_volume', 'volatility_pct'),
            ('volatility_pct', 'total_volume'),
            ('price_range', 'trade_count'),
            ('trade_count', 'price_range'),
            ('spread_pct', 'total_volume'),
            ('total_volume', 'spread_pct'),
            ('daily_return', 'total_volume'),
            ('total_volume', 'daily_return')
        ]
        
        # Tableau pour stocker les résultats
        granger_results = []
        
        # Effectuer les tests de causalité de Granger
        for var1, var2 in granger_pairs:
            data = pd.DataFrame({var1: df_corr[var1], var2: df_corr[var2]})
            max_lag = 7  # Tester jusqu'à 7 jours de décalage
            
            try:
                result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
                # Extraire la p-value minimale parmi tous les décalages
                min_p_value = min([result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)])
                best_lag = [i+1 for i in range(max_lag) if result[i+1][0]['ssr_chi2test'][1] == min_p_value][0]
                
                granger_results.append({
                    'cause': var1,
                    'effect': var2,
                    'min_p_value': min_p_value,
                    'best_lag': best_lag,
                    'significant': min_p_value < 0.05
                })
            except:
                print(f"Erreur lors du test de causalité entre {var1} et {var2}")
        
        # Afficher les résultats sous forme de tableau
        granger_df = pd.DataFrame(granger_results)
        print("\nRésultats des tests de causalité de Granger:")
        print(granger_df)
        
        # 4. Information mutuelle
        print("Calcul de l'information mutuelle...")
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculer l'information mutuelle entre chaque paire de variables
        mi_matrix = np.zeros((len(variables), len(variables)))
        
        for i, var1 in enumerate(variables):
            X = df_corr[variables].drop(var1, axis=1).values
            y = df_corr[var1].values
            mi = mutual_info_regression(X, y)
            
            for j, var2 in enumerate(variables):
                if var2 != var1:
                    mi_matrix[i, j] = mi[variables.index(var2) - (1 if variables.index(var2) > i else 0)]
        
        # Visualiser la matrice d'information mutuelle
        plt.figure(figsize=(14, 12))
        mi_df = pd.DataFrame(mi_matrix, index=variables, columns=variables)
        sns.heatmap(mi_df, cmap='viridis', annot=True, fmt=".3f")
        plt.title('Information mutuelle entre les variables du marché Bitcoin')
        
        if save_plots:
            plt.savefig(output_path / 'information_mutuelle.png', dpi=300)
        plt.show()
        
        # 5. Graphiques de dispersion conditionnelle
        print("Création de graphiques de dispersion conditionnelle...")
        
        # Définir les conditions de marché
        self.data_daily['market_condition'] = 'Normal'
        self.data_daily.loc[self.data_daily['volatility_7d'] > self.data_daily['volatility_7d'].quantile(0.75), 'market_condition'] = 'Haute volatilité'
        self.data_daily.loc[self.data_daily['volatility_7d'] < self.data_daily['volatility_7d'].quantile(0.25), 'market_condition'] = 'Basse volatilité'
        
        # Paires à visualiser
        scatter_pairs = [
            ('total_volume', 'volatility_pct'),
            ('price_range', 'trade_count'),
            ('spread_pct', 'total_volume')
        ]
        
        # Créer les graphiques de dispersion conditionnelle
        for var1, var2 in scatter_pairs:
            plt.figure(figsize=(15, 10))
            
            # Filtrer les données par condition de marché
            for condition, color, marker in zip(['Basse volatilité', 'Normal', 'Haute volatilité'], 
                                               ['blue', 'green', 'red'],
                                               ['o', 's', '^']):
                mask = self.data_daily['market_condition'] == condition
                data = self.data_daily[mask].dropna(subset=[var1, var2])
                
                plt.scatter(data[var1], data[var2], c=color, marker=marker, alpha=0.6, label=condition)
                
                # Ajouter une ligne de régression pour chaque condition
                if len(data) > 2:
                    z = np.polyfit(data[var1], data[var2], 1)
                    p = np.poly1d(z)
                    plt.plot(data[var1], p(data[var1]), c=color, linestyle='--')
            
            plt.title(f'Relation entre {var1} et {var2} selon les conditions de marché')
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.legend()
            plt.grid(True)
            
            if save_plots:
                plt.savefig(output_path / f'dispersion_{var1}_{var2}.png', dpi=300)
            plt.show()
        
        # 6. Analyse de la relation volume-volatilité dans le temps
        print("Analyse de l'évolution de la relation volume-volatilité...")
        
        # Calculer la corrélation glissante entre volume et volatilité
        window_sizes = [30, 90, 180]
        
        plt.figure(figsize=(15, 8))
        
        for window in window_sizes:
            # Calculer la corrélation glissante
            rolling_corr = self.data_daily['total_volume'].rolling(window=window).corr(self.data_daily['volatility_pct'])
            
            plt.plot(self.data_daily.index, rolling_corr, label=f'Fenêtre de {window} jours')
        
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Évolution de la corrélation entre volume et volatilité')
        plt.xlabel('Date')
        plt.ylabel('Corrélation')
        plt.legend()
        plt.grid(True)
        
        if save_plots:
            plt.savefig(output_path / 'correlation_glissante_volume_volatilite.png', dpi=300)
        plt.show()
        
        # 7. Analyse de l'interaction entre spread et volume
        print("Analyse de l'interaction entre spread et volume...")
        
        # Créer des bins de spread pour analyser le volume moyen par niveau de spread
        self.data_daily['spread_bin'] = pd.qcut(self.data_daily['spread_pct'], 10, labels=False)
        
        # Calculer le volume moyen par bin de spread
        spread_volume = self.data_daily.groupby('spread_bin')[['spread_pct', 'total_volume']].mean()
        spread_volume['spread_bin'] = spread_volume.index
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='spread_bin', y='total_volume', data=spread_volume)
        plt.title('Volume moyen par décile de spread')
        plt.xlabel('Décile de spread (0 = plus petit, 9 = plus grand)')
        plt.ylabel('Volume moyen')
        plt.grid(True, axis='y')
        
        if save_plots:
            plt.savefig(output_path / 'volume_par_spread.png', dpi=300)
        plt.show()
        
        # Retourner les résultats des analyses
        return {
            'correlation_matrix': corr_matrix,
            'granger_results': granger_df,
            'variables': variables
        }

    def analyse_volatilite_avancee(self, save_plots=False, output_dir='plots'):
        """Analyse approfondie de la volatilité avec des modèles avancés"""
        print("Analyse approfondie de la volatilité...")
        
        # Préparation des données
        returns = self.data_daily['daily_return'].dropna() / 100
        
        # 1. Modèle GARCH
        print("Modélisation GARCH de la volatilité...")
        model = arch_model(returns, vol='Garch', p=1, q=1)
        results = model.fit(disp='off')
        
        # Extraire la volatilité conditionnelle
        conditional_vol = results.conditional_volatility * np.sqrt(252) * 100
        
        # Visualiser la volatilité conditionnelle
        plt.figure(figsize=(15, 8))
        plt.plot(returns.index, conditional_vol, label='Volatilité conditionnelle (GARCH)')
        plt.title('Volatilité conditionnelle estimée par GARCH(1,1)')
        plt.xlabel('Date')
        plt.ylabel('Volatilité annualisée (%)')
        plt.grid(True)
        plt.legend()
        
        if save_plots:
            plt.savefig(Path(output_dir) / 'volatilite_garch.png', dpi=300)
        plt.show()
        
        # 2. Analyse de l'asymétrie de la volatilité
        print("Analyse de l'asymétrie de la volatilité...")
        
        # Séparer les rendements positifs et négatifs
        pos_returns = returns[returns > 0]
        neg_returns = returns[returns < 0]
        
        # Calculer la volatilité conditionnelle pour chaque type
        vol_after_pos = returns.shift(-1)[pos_returns.index].abs().mean() * np.sqrt(252) * 100
        vol_after_neg = returns.shift(-1)[neg_returns.index].abs().mean() * np.sqrt(252) * 100
        
        # Visualiser l'asymétrie
        plt.figure(figsize=(10, 6))
        bars = plt.bar(['Après rendements positifs', 'Après rendements négatifs'], 
                       [vol_after_pos, vol_after_neg])
        plt.title('Asymétrie de la volatilité')
        plt.ylabel('Volatilité moyenne suivante (%)')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        if save_plots:
            plt.savefig(Path(output_dir) / 'asymetrie_volatilite.png', dpi=300)
        plt.show()
        
        # 3. Clusters de volatilité
        print("Identification des clusters de volatilité...")
        
        # Préparer les données pour le clustering
        X = pd.DataFrame({
            'volatility': conditional_vol,
            'return_abs': returns.abs()
        })
        
        # Appliquer K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        X['cluster'] = kmeans.fit_predict(X)
        
        # Visualiser les clusters
        plt.figure(figsize=(15, 8))
        for i in range(3):
            mask = X['cluster'] == i
            plt.scatter(X.index[mask], X['volatility'][mask], 
                       label=f'Cluster {i}', alpha=0.6)
        
        plt.title('Clusters de volatilité')
        plt.xlabel('Date')
        plt.ylabel('Volatilité (%)')
        plt.legend()
        plt.grid(True)
        
        if save_plots:
            plt.savefig(Path(output_dir) / 'clusters_volatilite.png', dpi=300)
        plt.show()
        
        return {
            'garch_results': results,
            'volatility_clusters': X['cluster'],
            'conditional_volatility': conditional_vol
        }

    def generer_features_techniques(self, save_plots=False, output_dir='plots'):
        """Génère des indicateurs techniques avancés et des features d'ingénierie"""
        print("Génération des features techniques...")
        
        # 1. Indicateurs de momentum
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        
        # Calculer RSI
        self.data_daily['RSI'] = calculate_rsi(self.data_daily['close_price'])
        
        # Calculer MACD
        macd, signal = calculate_macd(self.data_daily['close_price'])
        self.data_daily['MACD'] = macd
        self.data_daily['MACD_signal'] = signal
        self.data_daily['MACD_hist'] = macd - signal
        
        # 2. Moyennes mobiles avancées
        windows = [5, 10, 20, 50, 100, 200]
        
        # EMA
        for window in windows:
            self.data_daily[f'EMA_{window}'] = self.data_daily['close_price'].ewm(span=window).mean()
        
        # KAMA (Kaufman Adaptive Moving Average)
        def calculate_kama(prices, er_period=10, fast_period=2, slow_period=30):
            change = abs(prices - prices.shift(er_period))
            volatility = abs(prices.diff()).rolling(er_period).sum()
            er = change / volatility
            fast_alpha = 2 / (fast_period + 1)
            slow_alpha = 2 / (slow_period + 1)
            sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            kama = prices.copy()
            for i in range(er_period, len(prices)):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
            return kama
        
        self.data_daily['KAMA'] = calculate_kama(self.data_daily['close_price'])
        
        # 3. Features de volume
        # Volume relatif
        for window in [5, 10, 20]:
            self.data_daily[f'volume_ratio_{window}'] = (
                self.data_daily['total_volume'] / 
                self.data_daily['total_volume'].rolling(window=window).mean()
            )
        
        # OBV (On-Balance Volume)
        self.data_daily['OBV'] = (self.data_daily['total_volume'] * 
                                 np.sign(self.data_daily['close_price'].diff())).cumsum()
        
        # 4. Features calendaires
        self.data_daily['day_of_week'] = self.data_daily.index.dayofweek
        self.data_daily['month'] = self.data_daily.index.month
        self.data_daily['quarter'] = self.data_daily.index.quarter
        
        # Variables cycliques
        self.data_daily['day_sin'] = np.sin(2 * np.pi * self.data_daily['day_of_week'] / 7)
        self.data_daily['day_cos'] = np.cos(2 * np.pi * self.data_daily['day_of_week'] / 7)
        self.data_daily['month_sin'] = np.sin(2 * np.pi * self.data_daily['month'] / 12)
        self.data_daily['month_cos'] = np.cos(2 * np.pi * self.data_daily['month'] / 12)
        
        # 5. Indicateurs de tendance
        def calculate_adx(high, low, close, period=14):
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = abs(100 * (minus_dm.rolling(window=period).mean() / atr))
            dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
            adx = dx.rolling(window=period).mean()
            
            return adx, plus_di, minus_di
        
        adx, plus_di, minus_di = calculate_adx(
            self.data_daily['max_price'],
            self.data_daily['min_price'],
            self.data_daily['close_price']
        )
        
        self.data_daily['ADX'] = adx
        self.data_daily['Plus_DI'] = plus_di
        self.data_daily['Minus_DI'] = minus_di
        
        # Visualiser les principaux indicateurs
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        
        # Prix et moyennes mobiles
        axes[0].plot(self.data_daily.index, self.data_daily['close_price'], label='Prix')
        axes[0].plot(self.data_daily.index, self.data_daily['EMA_20'], label='EMA 20')
        axes[0].plot(self.data_daily.index, self.data_daily['KAMA'], label='KAMA')
        axes[0].set_title('Prix et moyennes mobiles')
        axes[0].legend()
        axes[0].grid(True)
        
        # Indicateurs de momentum
        axes[1].plot(self.data_daily.index, self.data_daily['RSI'], label='RSI')
        axes[1].plot(self.data_daily.index, self.data_daily['MACD'], label='MACD')
        axes[1].plot(self.data_daily.index, self.data_daily['MACD_signal'], label='Signal MACD')
        axes[1].set_title('Indicateurs de momentum')
        axes[1].legend()
        axes[1].grid(True)
        
        # Volume et OBV
        axes[2].plot(self.data_daily.index, self.data_daily['volume_ratio_20'], label='Ratio Volume 20j')
        axes[2].plot(self.data_daily.index, self.data_daily['OBV'] / self.data_daily['OBV'].max(), 
                     label='OBV normalisé')
        axes[2].set_title('Indicateurs de volume')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(Path(output_dir) / 'indicateurs_techniques.png', dpi=300)
        plt.show()
        
        return self.data_daily


def main():
    predictor = BTCPredictor()
    
    # Chargement des données complètes (ajustez le limite si besoin pour tester)
    print("Chargement des données...")
    data = predictor.load_data(limit=None)
    print(f"Données chargées: {len(data)} entrées de {data.index.min()} à {data.index.max()}")
    
    # Agrégation des données à différentes échelles temporelles
    print("\nAgrégation des données...")
    aggregated = predictor.aggregate_data()
    
    # Analyse de la distribution et de l'évolution temporelle
    print("\nAnalyse de la distribution et de l'évolution temporelle...")
    #analysis_results = predictor.analyse_distribution_evolution(save_plots=True)
    
    # Analyse des tendances saisonnières et des cycles
    print("\nAnalyse des tendances saisonnières et des cycles...")
    #seasonal_analysis = predictor.analyse_tendances_saisonnieres(save_plots=True)
    
    # Analyse des corrélations entre variables
    print("\nAnalyse des corrélations entre variables...")
    #correlation_analysis = predictor.analyse_correlations_avancees(save_plots=True)
    
    # Analyse de la volatilité avancée
    print("\nAnalyse de la volatilité avancée...")
    volatility_analysis = predictor.analyse_volatilite_avancee(save_plots=True)
    
    # Génération des features techniques
    print("\nGénération des features techniques...")
    technical_features = predictor.generer_features_techniques(save_plots=True)
    
    print("\nAnalyse terminée!")


if __name__ == "__main__":
    main()