import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import sys
from pathlib import Path

# Ajout du chemin parent
sys.path.append(str(Path(__file__).parent.parent))

# Import avec le nouveau chemin
from features.data_processing import DataProcessing

# Initialisation des listes pour stocker les données
@st.cache_data
def initialize_data():
    return {
        'times': [],
        'prices': [],
        'pred_times': [],
        'predictions': []
    }

# Optimisation 1: Mise en cache du processeur de données
@st.cache_resource
def get_data_processor():
    return DataProcessing()

# Optimisation 2: Mise en cache de la création du graphique
@st.cache_data(ttl=1)
def create_chart(times, prices, pred_times, predictions):
    fig = go.Figure()
    
    # Ligne du prix réel (open_price)
    fig.add_trace(go.Scatter(
        x=times,
        y=prices,
        name='Prix d\'ouverture (temps réel)',
        line=dict(
            color='#00FF7F',
            width=1.5
        ),
        fill='tonexty',
        fillcolor='rgba(0, 255, 127, 0.1)',
        mode='lines'
    ))
    
    # Ligne de prédiction avec sa propre timeline
    fig.add_trace(go.Scatter(
        x=pred_times,
        y=predictions,
        name='Prédiction clôture journalière',
        line=dict(
            color='#FF4444',
            width=1.5
        ),
        mode='lines'
    ))
    
    # Configuration du layout avec titre plus précis
    fig.update_layout(
        title={
            'text': 'Bitcoin/USD - Prédiction de clôture journalière',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis_title='Prix (USD)',
        height=800,
        template='plotly_dark',
        plot_bgcolor='rgb(19, 23, 34)',  # Fond sombre comme dans l'image
        paper_bgcolor='rgb(19, 23, 34)',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(19, 23, 34, 0.95)'
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Configuration des axes avec nouveaux intervalles
    fig.update_xaxes(
        gridcolor='rgba(128, 128, 128, 0.1)',
        rangeslider_visible=False,
        dtick=1800000,  # 30 minutes (30 * 60 * 1000 millisecondes)
        tickformat='%H:%M',
        showgrid=True,
        tickmode='linear'  # Force l'utilisation d'intervalles linéaires
    )
    
    fig.update_yaxes(
        gridcolor='rgba(128, 128, 128, 0.1)',
        showgrid=True,
        tickformat=',.0f',  # Format avec séparateurs de milliers
        side='right',
        dtick=4000,  # Intervalle de 4000 USD
        tickmode='linear'  # Force l'utilisation d'intervalles linéaires
    )
    
    return fig

def main():
    # Optimisation 4: Configuration de la page en mode large par défaut
    st.set_page_config(
        page_title="Bitcoin Trading Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed"  # Cache la barre latérale
    )
    
    # Optimisation 5: Utilisation de containers pour réduire les rerenders
    with st.container():
        st.title("📈 Dashboard Trading Bitcoin")
        chart_placeholder = st.empty()

    # Initialisation optimisée
    data = initialize_data()
    data_processor = get_data_processor()  # Utilisation du cache
    # csv_path = "src/websocket/data/minute_data.csv"
    csv_path = "C:/Users/ouchaou/Desktop/ML/src/websocket/data/minute_data.csv" 

    try:
        for df_batch in data_processor.monitor_csv_file(csv_path):
            processed_df, last_row = data_processor.process_data(df_batch)
            
            # Mise à jour des données prix réel
            current_time = datetime.strptime(last_row['human_time'], '%Y-%m-%d %H:%M:%S')
            data['times'].append(current_time)
            data['prices'].append(last_row['open_price'])  # Utilisation de open_price
            
            # Optimisation 6: Mise en cache de la requête API
            @st.cache_data(ttl=1)
            def get_prediction(params):
                return requests.get("http://localhost:8000/predict", params=params).json()
            
            prediction_response = get_prediction({
                "open_price": last_row['open_price'],
                "max_price": last_row['max_price'],
                "min_price": last_row['min_price'],
                "EMA_50": last_row['EMA_50'],
                "BB_upper": last_row['BB_upper'],
                "human_time": last_row['human_time']
            })
            
            # Ajout de la prédiction avec sa date
            pred_time = datetime.strptime(prediction_response['datetime'], '%Y-%m-%d %H:%M:%S')
            data['pred_times'].append(pred_time)
            data['predictions'].append(prediction_response['predicted_close'])
            
            # Limitation à 24h de données
            if len(data['times']) > 1440:
                data['times'] = data['times'][-1440:]
                data['prices'] = data['prices'][-1440:]
                data['pred_times'] = data['pred_times'][-1440:]
                data['predictions'] = data['predictions'][-1440:]
            
            # Mise à jour du graphique
            with chart_placeholder:
                fig = create_chart(
                    data['times'], 
                    data['prices'],
                    data['pred_times'],  # Passage des dates de prédiction
                    data['predictions']
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
