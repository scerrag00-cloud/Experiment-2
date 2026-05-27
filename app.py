import streamlit as str
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import datetime
import io
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURAZIONE PAGINA ---
str.set_page_config(page_title="Super-Radar S&P 500 Multivariato", layout="wide")
str.title("🧠 Radar Predittivo S&P 500 - Modello Multivariato (15 Anni)")
str.markdown("Architettura Quantitativa Ibrida: Analisi NLP in tempo reale unita a 15 anni di dati Intermarket e Macroeconomici (FRED).")

# --- PANNELLO LATERALE ---
str.sidebar.header("1. Autenticazione")
api_key = str.sidebar.text_input("Inserisci NewsAPI Key:", type="password")
fred_api_key = str.sidebar.text_input("Inserisci FRED API Key:", type="password")

str.sidebar.header("2. Parametri di Rischio")
costo_fee = str.sidebar.number_input("Commissioni + Slippage % per Trade:", min_value=0.0, max_value=0.5, value=0.05, step=0.01) / 100
soglia_confidenza = str.sidebar.slider("Soglia Confidenza Algoritmica per Innesco:", min_value=0.51, max_value=0.65, value=0.54, step=0.01)

ambiti = {
    "Politica Monetaria": "(\"Federal Reserve\" OR \"interest rates\" OR \"inflation\") AND economy",
    "Dati Macroeconomici": "(\"US GDP\" OR \"unemployment rate\") AND economy",
    "Corporate & Innovazione": "(\"corporate earnings\" OR \"tech sector\") AND stocks",
    "Geopolitica & Crisi": "(geopolitics OR sanctions OR \"trade war\") AND NOT sports"
}

@str.cache_resource
def carica_modello_nlp():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analizza_notizie(api_key, query, nlp):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=4&apiKey={api_key}"
    risposta = requests.get(url).json()
    if risposta.get('status') != 'ok': return 0.0, ["Errore download notizie."]
    articles = risposta.get('articles', [])
    titoli = [art['title'] for art in articles]
    if not titoli: return 0.0, ["Nessun titolo rilevante trovato."]
    punteggio = sum([1 if nlp(t)[0]['label'] == 'positive' else -1 if nlp(t)[0]['label'] == 'negative' else 0 for t in titoli])
    return (punteggio / len(titoli)), titoli

if api_key:
    nlp = carica_modello_nlp()
    
    if str.button("Esegui Analisi Multivariata Storica"):
        with str.spinner("Estrazione dati macroeconomici FRED e storici di mercato (15 anni)..."):
            
            # 1. SCANSIONE NLP
            col1, col2, col3, col4 = str.columns(4)
            punteggi_oggi = {}
            for i, (nome, query) in enumerate(ambiti.items()):
                score, top_news = analizza_notizie(api_key, query, nlp)
                punteggi_oggi[nome] = score
                with [col1, col2, col3, col4][i]:
                    str.subheader(nome)
                    str.metric("Sentiment Odierno", f"{score:.2f}")
            
            # 2. DATA ENGINEERING: RETRIEVAL 15 ANNI
            inizio = "2011-01-01"
            fine = datetime.date.today().strftime("%Y-%m-%d")
            
            tickers = {
                'S&P 500': '^GSPC', 'Volatilità (VIX)': '^VIX', 
                'Tassi 10Y (TNX)': '^TNX', 'Nasdaq (IXIC)': '^IXIC',
                'Oro': 'GC=F', 'Petrolio': 'CL=F', 'Dollaro Index': 'UUP'
            }
            dati_yf = yf.download(list(tickers.values()), start=inizio, end=fine, progress=False)['Close']
            dati_yf = dati_yf.rename(columns={v: k for k, v in tickers.items()})
            
            # BLOCCO FRED UFFICIALE TRAMITE API
            try:
                if fred_api_key:
                    def preleva_fred(serie, chiave):
                        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={serie}&api_key={chiave}&file_type=json"
                        risposta = requests.get(url).json()
                        df = pd.DataFrame(risposta['observations'])
                        df['date'] = pd.to_datetime(df['date'])
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        return df.set_index('date')['value'].rename(serie)

                    df_unrate = preleva_fred('UNRATE', fred_api_key)
                    df_cpi = preleva_fred('CPIAUCSL', fred_api_key)
                    df_macro = pd.concat([df_unrate, df_cpi], axis=1)
                else:
                    str.warning("Inserisci la FRED API Key a sinistra per scaricare i dati macroeconomici reali.")
                    df_macro = pd.DataFrame({'UNRATE': 5.0, 'CPIAUCSL': 250.0}, index=dati_yf.index)
            except Exception as e:
                str.warning(f"Errore connessione FRED. Uso valori costanti. Dettaglio: {e}")
                df_macro = pd.DataFrame({'UNRATE': 5.0, 'CPIAUCSL': 250.0}, index=dati_yf.index)
            
            # Unione Dati
            df_totale = dati_yf.join(df_macro, how='left').ffill().dropna()
            
            # Creazione Features
            df_features = pd.DataFrame(index=df_totale.index)
            df_features['Rendimento_S&P500'] = df_totale['S&P 500'].pct_change()
            df_features['Variazione_VIX'] = df_totale['Volatilità (VIX)'].pct_change()
            df_features['Variazione_Tassi10Y'] = df_totale['Tassi 10Y (TNX)'].pct_change()
            df_features['Performance_Tech'] = df_totale['Nasdaq (IXIC)'].pct_change() - df_features['Rendimento_S&P500']
            df_features['Rendimento_Oro'] = df_totale['Oro'].pct_change()
            df_features['Rendimento_Petrolio'] = df_totale['Petrolio'].pct_change()
            df_features['Forza_Dollaro'] = df_totale['Dollaro Index'].pct_change()
            df_features['Trend_Disoccupazione'] = df_totale['UNRATE'].diff()
            df_features['Trend_Inflazione'] = df_totale['CPIAUCSL'].pct_change(12) 
            
            df_features['Politica Monetaria'] = -df_features['Variazione_Tassi10Y']
            df_features['Dati Macroeconomici'] = df_features['Rendimento_S&P500']
            df_features['Corporate & Innovazione'] = df_features['Performance_Tech']
            df_features['Geopolitica & Crisi'] = -df_features['Variazione_VIX']
            
            df_features['Target'] = (df_features['Rendimento_S&P500'].shift(-1) > 0).astype(int)
            df_features = df_features.dropna()
            
            # 3. MACHINE LEARNING MULTIVARIATO
            lista_predittori = ['Politica Monetaria', 'Dati Macroeconomici', 'Corporate & Innovazione', 'Geopolitica & Crisi', 
                                'Rendimento_Oro', 'Rendimento_Petrolio', 'Forza_Dollaro', 'Trend_Disoccupazione', 'Trend_Inflazione']
            
            X = df_features[lista_predittori]
            y = df_features['Target']
            
            modello = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
            modello.fit(X, y)
            
            ultimo_giorno = df_features.iloc[-1]
            dati_oggi = pd.DataFrame([{
                'Politica Monetaria': punteggi_oggi['Politica Monetaria'],
                'Dati Macroeconomici': punteggi_oggi['Dati Macroeconomici'],
                'Corporate & Innovazione': punteggi_oggi['Corporate & Innovazione'],
                'Geopolitica & Crisi': punteggi_oggi['Geopolitica & Crisi'],
                'Rendimento_Oro': ultimo_giorno['Rendimento_Oro'],
                'Rendimento_Petrolio': ultimo_giorno['Rendimento_Petrolio'],
                'Forza_Dollaro': ultimo_giorno['Forza_Dollaro'],
                'Trend_Disoccupazione': ultimo_giorno['Trend_Disoccupazione'],
                'Trend_Inflazione': ultimo_giorno['Trend_Inflazione']
            }])
            
            previsione = modello.predict(dati_oggi)[0]
            probabilita = modello.predict_proba(dati_oggi)[0][previsione] * 100
            
            testo_direzione = "RIALZO 📈" if previsione == 1 else "RIBASSO 📉"
            
            str.markdown("---")
            str.header("🔮 VERDETTO DELL'INTELLIGENZA ARTIFICIALE MULTIVARIATA")
            mc1, mc2 = str.columns(2)
            mc1.metric("Proiezione Statistica Prossima Sessione", testo_direzione)
            mc2.metric("Confidenza dell'Insieme Alberi", f"{probabilita:.1f}%")
            
            # 4. ENGINE DI BACKTESTING RIGOROSO
            str.markdown("---")
            str.header("📈 Validazione Storica della Strategia (Dal 2011 a Oggi)")
            
            df_features['Prob_Rialzo'] = modello.predict_proba(X)[:, 1]
            df_features['Volatilita_Attesa'] = df_features['Rendimento_S&P500'].rolling(15).std().fillna(0)
            df_features['Segnale'] = np.where((df_features['Prob_Rialzo'] > soglia_confidenza) & (df_features['Volatilita_Attesa'] > (costo_fee * 2)), 1, 0)
            
            df_features['Rendimento_Mercato'] = df_features['Rendimento_S&P500'].shift(-1)
            df_features['Rendimento_Strategia'] = df_features['Segnale'] * df_features['Rendimento_Mercato']
            
            df_features['Variazione_Posizione'] = df_features['Segnale'].diff().abs()
            df_features.loc[df_features['Variazione_Posizione'] == 1, 'Rendimento_Strategia'] -= costo_fee
            
            capitale_iniziale = 10000
            df_features['S&P 500 (Buy & Hold)'] = capitale_iniziale * (1 + df_features['Rendimento_Mercato']).cumprod()
            df_features['Algoritmo Multivariato Netto'] = capitale_iniziale * (1 + df_features['Rendimento_Strategia']).cumprod()
            
            df_plot = df_features[['Algoritmo Multivariato Netto', 'S&P 500 (Buy & Hold)']].dropna()
            
            rend_b_h = ((df_plot['S&P 500 (Buy & Hold)'].iloc[-1] / capitale_iniziale) - 1) * 100
            rend_ai = ((df_plot['Algoritmo Multivariato Netto'].iloc[-1] / capitale_iniziale) - 1) * 100
            trade_totali = df_features['Variazione_Posizione'].sum() / 2
            
            rc1, rc2, rc3 = str.columns(3)
            rc1.metric("Rendimento Storico Indice", f"{rend_b_h:.1f}%")
            rc2.metric("Rendimento Netto Modello AI", f"{rend_ai:.1f}%", delta=f"{rend_ai - rend_b_h:.1f}% vs Benchmark")
            rc3.metric("Operazioni Totali Eseguite", f"{int(trade_totali)}")
            
            str.line_chart(df_plot)
            
            # 5. IMPORTANZA DELLE VARIABILI
            str.markdown("---")
            str.subheader("📊 Analisi dell'Importanza dei Fattori (Feature Importance)")
            importanza = pd.DataFrame({'Fattore': lista_predittori, 'Importanza': modello.feature_importances_})
            importanza = importanza.sort_values(by='Importanza', ascending=False).set_index('Fattore')
            str.bar_chart(importanza)
else:
    str.warning("Inserisci la chiave API nella barra laterale per sbloccare i predittori multivariati.")
