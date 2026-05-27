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
str.set_page_config(page_title="Super-Radar S&P 500", layout="wide")
str.title("🧠 Radar Predittivo S&P 500 - Swing & News Adattivo")
str.markdown("Architettura Ibrida: Previsione a 5 Giorni, Riadattamento Continuo (Walk-Forward) e Notifiche Smart.")

# --- PANNELLO LATERALE ---
str.sidebar.header("1. Autenticazione API")
api_key = str.sidebar.text_input("NewsAPI Key:", type="password")
fred_api_key = str.sidebar.text_input("FRED API Key (Opzionale):", type="password")

str.sidebar.header("2. Telegram Bot")
TELEGRAM_TOKEN = str.sidebar.text_input("Bot Token:", type="password")
TELEGRAM_CHAT_ID = str.sidebar.text_input("Chat ID:", type="password")

str.sidebar.header("3. Parametri Operativi")
costo_fee = str.sidebar.number_input("Commissioni + Slippage %:", min_value=0.0, max_value=0.5, value=0.05, step=0.01) / 100
soglia_confidenza = str.sidebar.slider("Soglia Confidenza AI:", min_value=0.51, max_value=0.65, value=0.54, step=0.01)
orizzonte_giorni = str.sidebar.slider("Orizzonte (Giorni):", min_value=1, max_value=10, value=5, step=1)

ambiti = {
    "Politica Monetaria": "(\"Federal Reserve\" OR \"interest rates\" OR \"inflation\") AND economy",
    "Dati Macroeconomici": "(\"US GDP\" OR \"unemployment rate\") AND economy",
    "Corporate & Innovazione": "(\"corporate earnings\" OR \"tech sector\") AND stocks",
    "Geopolitica & Crisi": "(geopolitics OR sanctions OR \"trade war\") AND NOT sports"
}

@str.cache_resource
def carica_modello_nlp():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analizza_notizie(chiave_api, query, nlp_model):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=4&apiKey={chiave_api}"
    try:
        risposta = requests.get(url).json()
        if risposta.get('status') != 'ok': return 0.0, ["Errore download notizie."]
        articles = risposta.get('articles', [])
        titoli = [art['title'] for art in articles if art.get('title')]
        if not titoli: return 0.0, ["Nessun titolo rilevante."]
        
        punteggio = sum([1 if nlp_model(t)[0]['label'] == 'positive' else -1 if nlp_model(t)[0]['label'] == 'negative' else 0 for t in titoli])
        return (punteggio / len(titoli)), titoli
    except Exception:
        return 0.0, ["Errore di connessione API."]

def invia_messaggio_telegram(testo):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": testo, "parse_mode": "Markdown"}
    try: 
        requests.post(url, json=payload)
    except Exception as e: 
        str.error(f"Errore invio Telegram: {e}")

if api_key:
    nlp = carica_modello_nlp()
    
    if str.button("Esegui Analisi Completa"):
        with str.spinner(f"Elaborazione dati, lettura news e calcolo walk-forward a {orizzonte_giorni} giorni..."):
            
            # 1. SCANSIONE NLP E SALVATAGGIO NEWS
            col1, col2, col3, col4 = str.columns(4)
            punteggi_oggi = {}
            top_news_memoria = {} 
            
            for i, (nome, query) in enumerate(ambiti.items()):
                score, top_news = analizza_notizie(api_key, query, nlp)
                punteggi_oggi[nome] = score
                top_news_memoria[nome] = top_news
                with [col1, col2, col3, col4][i]:
                    str.subheader(nome)
                    str.metric("Sentiment", f"{score:.2f}")
            
            # 2. DATA ENGINEERING STORICO E MACROECONOMICO
            inizio = "2011-01-01"
            fine = datetime.date.today().strftime("%Y-%m-%d")
            
            tickers = {'S&P 500': '^GSPC', 'Volatilità (VIX)': '^VIX', 'Tassi 10Y (TNX)': '^TNX', 'Nasdaq (IXIC)': '^IXIC', 'Oro': 'GC=F', 'Petrolio': 'CL=F', 'Dollaro Index': 'UUP'}
            dati_yf = yf.download(list(tickers.values()), start=inizio, end=fine, progress=False)['Close']
            dati_yf = dati_yf.rename(columns={v: k for k, v in tickers.items()})
            
            try:
                if fred_api_key:
                    def preleva_fred(serie, chiave):
                        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={serie}&api_key={chiave}&file_type=json"
                        risposta = requests.get(url).json()
                        df = pd.DataFrame(risposta['observations'])
                        df['date'] = pd.to_datetime(df['date'])
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        return df.set_index('date')['value'].rename(serie)
                    df_macro = pd.concat([preleva_fred('UNRATE', fred_api_key), preleva_fred('CPIAUCSL', fred_api_key)], axis=1)
                else:
                    df_macro = pd.DataFrame({'UNRATE': 5.0, 'CPIAUCSL': 250.0}, index=dati_yf.index)
            except Exception:
                df_macro = pd.DataFrame({'UNRATE': 5.0, 'CPIAUCSL': 250.0}, index=dati_yf.index)
            
            df_totale = dati_yf.join(df_macro, how='left').ffill().dropna()
            
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
            
            df_features[f'Target_{orizzonte_giorni}g'] = (df_totale['S&P 500'].pct_change(periods=orizzonte_giorni).shift(-orizzonte_giorni) > 0).astype(int)
            df_features = df_features.dropna()
            
            lista_predittori = ['Politica Monetaria', 'Dati Macroeconomici', 'Corporate & Innovazione', 'Geopolitica & Crisi', 'Rendimento_Oro', 'Rendimento_Petrolio', 'Forza_Dollaro', 'Trend_Disoccupazione', 'Trend_Inflazione']
            X = df_features[lista_predittori]
            y = df_features[f'Target_{orizzonte_giorni}g']
            
            # 3. BACKTEST WALK-FORWARD ADATTIVO
            str.markdown("---")
            str.header("📈 Validazione Walk-Forward (Test alla cieca - Ultimi 3 Anni)")
            indice_taglio = int(len(df_features) * 0.8)
            probabilita_storia = []
            passo = 21 
            
            progresso = str.progress(0, "Addestramento continuo della Memoria Dinamica in corso...")
            for i in range(indice_taglio, len(df_features), passo):
                modello_locale = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                modello_locale.fit(X.iloc[:i], y.iloc[:i])
                fine_test = min(i + passo, len(df_features))
                probabilita_storia.extend(modello_locale.predict_proba(X.iloc[i:fine_test])[:, 1])
                progresso.progress(min((i - indice_taglio) / (len(df_features) - indice_taglio), 1.0))
            progresso.empty()
            
            df_test = df_features.iloc[indice_taglio:indice_taglio+len(probabilita_storia)].copy()
            df_test['Prob'] = probabilita_storia
            df_test['Vol_Attesa'] = df_test['Rendimento_S&P500'].rolling(10).std().fillna(0)
            df_test['Segnale'] = np.where((df_test['Prob'] > soglia_confidenza) & (df_test['Vol_Attesa'] > costo_fee), 1, 0)
            df_test['Posizione'] = df_test['Segnale'].rolling(orizzonte_giorni, min_periods=1).max()
            df_test['Rend_Mercato'] = df_test['Rendimento_S&P500'].shift(-1)
            df_test['Rend_Strat'] = df_test['Posizione'] * df_test['Rend_Mercato']
            df_test['Var_Pos'] = df_test['Posizione'].diff().abs()
            df_test.loc[df_test['Var_Pos'] == 1, 'Rend_Strat'] -= costo_fee
            
            cap_iniziale = 10000
            df_test['Benchmark'] = cap_iniziale * (1 + df_test['Rend_Mercato']).cumprod()
            df_test['AI_Netto'] = cap_iniziale * (1 + df_test['Rend_Strat']).cumprod()
            df_plot = df_test[['AI_Netto', 'Benchmark']].dropna()
            
            rc1, rc2, rc3 = str.columns(3)
            rc1.metric("Mercato (3 Anni)", f"{((df_plot['Benchmark'].iloc[-1] / cap_iniziale) - 1)*100:.1f}%")
            rc2.metric("Strategia AI", f"{((df_plot['AI_Netto'].iloc[-1] / cap_iniziale) - 1)*100:.1f}%")
            rc3.metric("Trade Eseguiti (Buy/Sell)", f"{int(df_test['Var_Pos'].sum()/2)}")
            str.line_chart(df_plot)
            
            # 4. CALENDARIO, PREVISIONE FINALE E TELEGRAM
            modello_finale = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
            modello_finale.fit(X, y) 
            
            ultimo = df_features.iloc[-1]
            dati_oggi = pd.DataFrame([{
                'Politica Monetaria': punteggi_oggi['Politica Monetaria'], 'Dati Macroeconomici': punteggi_oggi['Dati Macroeconomici'],
                'Corporate & Innovazione': punteggi_oggi['Corporate & Innovazione'], 'Geopolitica & Crisi': punteggi_oggi['Geopolitica & Crisi'],
                'Rendimento_Oro': ultimo['Rendimento_Oro'], 'Rendimento_Petrolio': ultimo['Rendimento_Petrolio'],
                'Forza_Dollaro': ultimo['Forza_Dollaro'], 'Trend_Disoccupazione': ultimo['Trend_Disoccupazione'], 'Trend_Inflazione': ultimo['Trend_Inflazione']
            }])
            
            previsione = modello_finale.predict(dati_oggi)[0]
            probabilita = modello_finale.predict_proba(dati_oggi)[0][previsione] * 100
            
            # Gestione dei giorni (0 = Lunedì, 4 = Venerdì)
            oggi_index = datetime.datetime.today().weekday()
            giorno_target = "Lunedì" if oggi_index == 4 else "Domani"
            
            testo_direzione = f"RIALZO 📈" if previsione == 1 else f"RIBASSO 📉"
            
            str.markdown("---")
            str.header(f"🔮 VERDETTO OPERATIVO")
            mc1, mc2 = str.columns(2)
            mc1.metric(f"Trend Previso ({orizzonte_giorni}gg a partire da {giorno_target})", testo_direzione)
            mc2.metric("Confidenza", f"{probabilita:.1f}%")
            
            # Compilazione organica delle migliori notizie
            resoconto_news = ""
            for cat, news_list in top_news_memoria.items():
                if news_list and len(news_list) > 0 and news_list[0] != "Nessun titolo rilevante.":
                    resoconto_news += f"*{cat}*\n"
                    for n in news_list[:2]: 
                        resoconto_news += f"▪️ {n}\n"
                    resoconto_news += "\n"
            
            # Generazione e Invio Alert
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                msg = (
                    f"🤖 *Radar S&P 500 - Chiusura*\n\n"
                    f"🎯 *Trend Operativo ({orizzonte_giorni}gg da {giorno_target}):* {testo_direzione}\n"
                    f"📊 *Confidenza AI:* {probabilita:.1f}%\n\n"
                    f"📰 *SINTESI NEWS DI OGGI:*\n{resoconto_news}"
                    f"🌍 *Sentiment Globale:*\n"
                    f"• Monetario: {punteggi_oggi['Politica Monetaria']:.2f}\n"
                    f"• Macro: {punteggi_oggi['Dati Macroeconomici']:.2f}\n"
                    f"• Geopolitica: {punteggi_oggi['Geopolitica & Crisi']:.2f}"
                )
                invia_messaggio_telegram(msg)
                str.success(f"📲 Report dettagliato e Segnale per {giorno_target} inviati a Telegram!")
else:
    str.warning("Inserisci la NewsAPI Key nella barra laterale per sbloccare il sistema.")
