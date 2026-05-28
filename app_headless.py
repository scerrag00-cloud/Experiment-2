import os
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import datetime
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier

# --- CHIAVI SEGRETE ---
api_key = os.environ.get("NEWS_API_KEY")
fred_api_key = os.environ.get("FRED_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- PARAMETRI FISSI PER L'AUTOMAZIONE ---
costo_fee = 0.05 / 100
soglia_confidenza = 0.54
orizzonte_giorni = 5

ambiti = {
    "Politica Monetaria": "(\"Federal Reserve\" OR \"interest rates\" OR \"inflation\") AND economy",
    "Dati Macroeconomici": "(\"US GDP\" OR \"unemployment rate\") AND economy",
    "Corporate & Innovazione": "(\"corporate earnings\" OR \"tech sector\") AND stocks",
    "Geopolitica & Crisi": "(geopolitics OR sanctions OR \"trade war\") AND NOT sports"
}

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
    requests.post(url, json=payload)

if not api_key:
    print("ERRORE: NEWS_API_KEY mancante.")
    exit()

nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
punteggi_oggi = {}
top_news_memoria = {} 

for nome, query in ambiti.items():
    score, top_news = analizza_notizie(api_key, query, nlp)
    punteggi_oggi[nome] = score
    top_news_memoria[nome] = top_news

inizio = "2011-01-01"
fine = datetime.date.today().strftime("%Y-%m-%d")

tickers = {'S&P 500': '^GSPC', 'Volatilità (VIX)': '^VIX', 'Tassi 10Y (TNX)': '^TNX', 'Nasdaq (IXIC)': '^IXIC', 'Oro': 'GC=F', 'Petrolio': 'CL=F', 'Dollaro Index': 'UUP'}

# Aggiunto threads=False per evitare il blocco del database di yfinance
dati_yf = yf.download(list(tickers.values()), start=inizio, end=fine, progress=False, threads=False)['Close']
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

modello_finale = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
modello_finale.fit(X, y) 

# --- NUOVA SEZIONE: VERIFICA PREVISIONE PASSATA ---
indice_passato = -1 - orizzonte_giorni
features_passate = X.iloc[[indice_passato]]

pred_passata = modello_finale.predict(features_passate)[0]
prob_passata = modello_finale.predict_proba(features_passate)[0][pred_passata] * 100

direzione_passata = "RIALZO 📈" if pred_passata == 1 else "RIBASSO 📉"

prezzo_oggi = df_totale['S&P 500'].iloc[-1]
prezzo_passato = df_totale['S&P 500'].iloc[indice_passato]
ritorno_reale = ((prezzo_oggi / prezzo_passato) - 1) * 100

direzione_reale = "RIALZO 📈" if ritorno_reale > 0 else "RIBASSO 📉"

if (pred_passata == 1 and ritorno_reale > 0) or (pred_passata == 0 and ritorno_reale <= 0):
    esito_passato = "✅ GIUSTA"
else:
    esito_passato = "❌ ERRATA"
# --------------------------------------------------

# Previsione per il futuro
ultimo = df_features.iloc[-1]
dati_oggi = pd.DataFrame([{
    'Politica Monetaria': punteggi_oggi['Politica Monetaria'], 'Dati Macroeconomici': punteggi_oggi['Dati Macroeconomici'],
    'Corporate & Innovazione': punteggi_oggi['Corporate & Innovazione'], 'Geopolitica & Crisi': punteggi_oggi['Geopolitica & Crisi'],
    'Rendimento_Oro': ultimo['Rendimento_Oro'], 'Rendimento_Petrolio': ultimo['Rendimento_Petrolio'],
    'Forza_Dollaro': ultimo['Forza_Dollaro'], 'Trend_Disoccupazione': ultimo['Trend_Disoccupazione'], 'Trend_Inflazione': ultimo['Trend_Inflazione']
}])

previsione = modello_finale.predict(dati_oggi)[0]
probabilita = modello_finale.predict_proba(dati_oggi)[0][previsione] * 100

# --- LOGICA DEL SEMAFORO OPERATIVO ---
if probabilita >= (soglia_confidenza * 100):
    azione_consigliata = "🟢 INGRESSO CONSIGLIATO (Confidenza elevata)"
else:
    azione_consigliata = "🟡 RESTARE IN ATTESA (Segnale troppo debole)"

oggi_index = datetime.datetime.today().weekday()
giorno_target = "Lunedì" if oggi_index == 4 else "Domani"
testo_direzione = f"RIALZO 📈" if previsione == 1 else f"RIBASSO 📉"

resoconto_news = ""
for cat, news_list in top_news_memoria.items():
    if news_list and len(news_list) > 0 and news_list[0] != "Nessun titolo rilevante.":
        resoconto_news += f"*{cat}*\n"
        for n in news_list[:2]: 
            resoconto_news += f"▪️ {n}\n"
        resoconto_news += "\n"

msg = (
    f"🤖 *Radar S&P 500 - Report Notturno*\n\n"
    f"🔮 *NUOVA PREVISIONE:* \n"
    f"• Trend ({orizzonte_giorni}gg da {giorno_target}): *{testo_direzione}*\n"
    f"• Confidenza: *{probabilita:.1f}%*\n"
    f"⚠️ *Azione AI:* {azione_consigliata}\n\n"
    f"📊 *VERIFICA PREVISIONE PRECEDENTE:* \n"
    f"• Finestra temporale: ultimi {orizzonte_giorni} giorni di borsa\n"
    f"• Previsione dell'AI: {direzione_passata} (con {prob_passata:.1f}% confidenza)\n"
    f"• Movimento Reale: {direzione_reale} ({ritorno_reale:+.2f}%)\n"
    f"• Esito: *{esito_passato}*\n\n"
    f"📰 *SINTESI NEWS DI OGGI:*\n{resoconto_news}"
    f"🌍 *Sentiment Globale:*\n"
    f"• Monetario: {punteggi_oggi['Politica Monetaria']:.2f}\n"
    f"• Macro: {punteggi_oggi['Dati Macroeconomici']:.2f}\n"
    f"• Geopolitica: {punteggi_oggi['Geopolitica & Crisi']:.2f}"
)

invia_messaggio_telegram(msg)
print("Analisi e verifica completate. Messaggio inviato.")
