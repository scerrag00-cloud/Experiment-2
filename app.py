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
str.title("🧠 Radar Predittivo S&P 500 - Swing Trading & AI Adattiva")
str.markdown("Architettura Ibrida: Previsione a 5 Giorni (Swing Trading) con Riadattamento Continuo (Walk-Forward).")

# --- PANNELLO LATERALE ---
str.sidebar.header("1. Autenticazione")
api_key = str.sidebar.text_input("Inserisci NewsAPI Key:", type="password")
fred_api_key = str.sidebar.text_input("Inserisci FRED API Key (Opzionale):", type="password")

# --- TELEGRAM ---
str.sidebar.header("Telegram Bot (Opzionale)")
TELEGRAM_TOKEN = str.sidebar.text_input("Telegram Bot Token:", type="password")
TELEGRAM_CHAT_ID = str.sidebar.text_input("Telegram Chat ID:", type="password")

str.sidebar.header("2. Parametri Operativi")
costo_fee = str.sidebar.number_input("Commissioni + Slippage % per Trade:", min_value=0.0, max_value=0.5, value=0.05, step=0.01) / 100
soglia_confidenza = str.sidebar.slider("Soglia Confidenza AI per Ingresso:", min_value=0.51, max_value=0.65, value=0.54, step=0.01)
orizzonte_giorni = str.sidebar.slider("Orizzonte Previsione (Giorni):", min_value=1, max_value=10, value=5, step=1)

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

def invia_messaggio_telegram(testo):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": testo, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        str.error(f"Errore invio Telegram: {e}")

if api_key:
    nlp = carica_modello_nlp()
    
    if str.button("Esegui Analisi Avanzata"):
        with str.spinner(f"Estrazione dati storici e calcolo previsioni Swing a {orizzonte_giorni} giorni..."):
            
            # 1. SCANSIONE NLP IN TEMPO REALE
            col1, col2, col3, col4 = str.columns(4)
            punteggi_oggi = {}
            for i, (nome, query) in enumerate(ambiti.items()):
                score, top_news = analizza_notizie(api_key, query, nlp)
                punteggi_oggi[nome] = score
                with [col1, col2, col3, col4][i]:
                    str.subheader(nome)
                    str.metric("Sentiment Odierno", f"{score:.2f}")
            
            # 2. DATA ENGINEERING
            inizio = "2011-01-01"
            fine = datetime.date.today().strftime("%Y-%m-%d")
            
            tickers = {
                'S&P 500': '^GSPC', 'Volatilità (VIX)': '^VIX', 
                'Tassi 10Y (TNX)': '^TNX', 'Nasdaq (IXIC)': '^IXIC',
                'Oro': 'GC=F', 'Petrolio': 'CL=F', 'Dollaro Index': 'UUP'
            }
            dati_yf = yf.download(list(tickers.values()), start=inizio, end=fine, progress=False)['Close']
            dati_yf = dati_yf.rename(columns={v: k for k, v in tickers.items()})
            
            # BLOCCO FRED UFFICIALE
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
                    str.warning("Nessuna FRED API Key inserita. Uso valori macroeconomici costanti per la simulazione.")
                    df_macro = pd.DataFrame({'UNRATE': 5.0, 'CPIAUCSL': 250.0}, index=dati_yf.index)
            except Exception as e:
                str.warning(f"Errore connessione FRED. Uso valori costanti. Dettaglio: {e}")
                df_macro = pd.DataFrame({'UNRATE': 5.0, 'CPIAUCSL': 250.0}, index=dati_yf.index)
            
            # Unione Dati
            df_totale = dati_yf.join(df_macro, how='left').ffill().dropna()
            
            # Creazione Features
            df_features = pd.DataFrame(index=df_totale.index)
            df_features['Rendimento_S&P500_Giornaliero'] = df_totale['S&P 500'].pct_change()
            df_features['Variazione_VIX'] = df_totale['Volatilità (VIX)'].pct_change()
            df_features['Variazione_Tassi10Y'] = df_totale['Tassi 10Y (TNX)'].pct_change()
            df_features['Performance_Tech'] = df_totale['Nasdaq (IXIC)'].pct_change() - df_features['Rendimento_S&P500_Giornaliero']
            df_features['Rendimento_Oro'] = df_totale['Oro'].pct_change()
            df_features['Rendimento_Petrolio'] = df_totale['Petrolio'].pct_change()
            df_features['Forza_Dollaro'] = df_totale['Dollaro Index'].pct_change()
            df_features['Trend_Disoccupazione'] = df_totale['UNRATE'].diff()
            df_features['Trend_Inflazione'] = df_totale['CPIAUCSL'].pct_change(12) 
            
            df_features['Politica Monetaria'] = -df_features['Variazione_Tassi10Y']
            df_features['Dati Macroeconomici'] = df_features['Rendimento_S&P500_Giornaliero']
            df_features['Corporate & Innovazione'] = df_features['Performance_Tech']
            df_features['Geopolitica & Crisi'] = -df_features['Variazione_VIX']
            
            # --- MODIFICA 1: SWING TRADING TARGET ---
            # Chiediamo all'AI se il mercato sarà salito tra N giorni rispetto a oggi.
            df_features[f'Rendimento_Futuro_{orizzonte_giorni}g'] = df_totale['S&P 500'].pct_change(periods=orizzonte_giorni).shift(-orizzonte_giorni)
            df_features['Target'] = (df_features[f'Rendimento_Futuro_{orizzonte_giorni}g'] > 0).astype(int)
            
            df_features = df_features.dropna()
            
            lista_predittori = ['Politica Monetaria', 'Dati Macroeconomici', 'Corporate & Innovazione', 'Geopolitica & Crisi', 
                                'Rendimento_Oro', 'Rendimento_Petrolio', 'Forza_Dollaro', 'Trend_Disoccupazione', 'Trend_Inflazione']
            
            X = df_features[lista_predittori]
            y = df_features['Target']
            
            # --- MODIFICA 2: WALK-FORWARD (AI ADATTIVA) ---
            str.markdown("---")
            str.header("📈 Validazione Avanzata: Swing Trading Adattivo (Ultimi 3 Anni)")
            str.markdown(f"*L'Intelligenza Artificiale si è ri-addestrata mensilmente, imparando dalle crisi recenti, per prevedere cicli di {orizzonte_giorni} giorni.*")
            
            # Impostiamo il punto di inizio test a 3 anni fa
            indice_taglio_iniziale = int(len(df_features) * 0.8)
            mesi_di_test = len(df_features) - indice_taglio_iniziale
            passo_finestra = 21  # Riadattamento ogni mese lavorativo circa (21 giorni di borsa)
            
            probabilita_storia = []
            
            # Il ciclo che ri-addestra l'AI spostandosi in avanti nel tempo
            progresso = str.progress(0, text="Addestramento continuo della Memoria Dinamica in corso...")
            for i in range(indice_taglio_iniziale, len(df_features), passo_finestra):
                # Il modello vede sempre TUTTO dal 2011 fino al mese "corrente"
                X_train = X.iloc[:i]
                y_train = y.iloc[:i]
                
                # Prepara la finestra di test (il mese successivo)
                fine_test = min(i + passo_finestra, len(df_features))
                X_test = X.iloc[i:fine_test]
                
                modello_locale = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                modello_locale.fit(X_train, y_train)
                
                # Prevede i giorni della finestra di test
                prob = modello_locale.predict_proba(X_test)[:, 1]
                probabilita_storia.extend(prob)
                
                # Aggiorna la barra di caricamento
                progresso.progress(min((i - indice_taglio_iniziale) / mesi_di_test, 1.0))
            
            progresso.empty()
            
            # Creazione del DataFrame di Test finale unendo le previsioni adattive
            df_test = df_features.iloc[indice_taglio_iniziale:indice_taglio_iniziale+len(probabilita_storia)].copy()
            df_test['Prob_Rialzo'] = probabilita_storia
            
            # --- MODIFICA 3: LOGICA DI INGRESSO SWING ---
            df_test['Volatilita_Attesa'] = df_test['Rendimento_S&P500_Giornaliero'].rolling(10).std().fillna(0)
            
            # Generazione del Segnale
            df_test['Segnale_Generato'] = np.where((df_test['Prob_Rialzo'] > soglia_confidenza) & (df_test['Volatilita_Attesa'] > (costo_fee)), 1, 0)
            
            # Mantieni la posizione per i giorni previsti dall'orizzonte (es. 5 giorni) senza uscire e rientrare
            df_test['Posizione_Attiva'] = df_test['Segnale_Generato'].rolling(window=orizzonte_giorni, min_periods=1).max()
            
            # Calcolo Rendimenti
            df_test['Rendimento_Mercato'] = df_test['Rendimento_S&P500_Giornaliero'].shift(-1)
            df_test['Rendimento_Strategia'] = df_test['Posizione_Attiva'] * df_test['Rendimento_Mercato']
            
            # Sottrazione analitica dei costi di transazione (Solo quando ENTRI o ESCI davvero dalla posizione a 5 giorni)
            df_test['Variazione_Posizione'] = df_test['Posizione_Attiva'].diff().abs()
            df_test.loc[df_test['Variazione_Posizione'] == 1, 'Rendimento_Strategia'] -= costo_fee
            
            capitale_iniziale = 10000
            df_test['S&P 500 (Buy & Hold)'] = capitale_iniziale * (1 + df_test['Rendimento_Mercato']).cumprod()
            df_test['AI Swing Adattivo Netto'] = capitale_iniziale * (1 + df_test['Rendimento_Strategia']).cumprod()
            
            df_plot = df_test[['AI Swing Adattivo Netto', 'S&P 500 (Buy & Hold)']].dropna()
            
            rend_b_h = ((df_plot['S&P 500 (Buy & Hold)'].iloc[-1] / capitale_iniziale) - 1) * 100
            rend_ai = ((df_plot['AI Swing Adattivo Netto'].iloc[-1] / capitale_iniziale) - 1) * 100
            trade_totali = df_test['Variazione_Posizione'].sum() / 2
            
            rc1, rc2, rc3 = str.columns(3)
            rc1.metric("Rendimento Mercato (Test 3 Anni)", f"{rend_b_h:.1f}%")
            rc2.metric("Rendimento Netto Strategia Swing", f"{rend_ai:.1f}%", delta=f"{rend_ai - rend_b_h:.1f}% vs Benchmark")
            rc3.metric("Operazioni Totali Eseguite (Buy/Sell)", f"{int(trade_totali)}")
            
            str.line_chart(df_plot)
            
            # 3. PREVISIONE PER LA SETTIMANA ENTRANTE (Addestramento Finale su Tutti i Dati)
            modello_finale = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
            modello_finale.fit(X, y) # Ora studia tutto fino a IERI per prevedere DOMANI
            
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
            
            previsione = modello_finale.predict(dati_oggi)[0]
            probabilita = modello_finale.predict_proba(dati_oggi)[0][previsione] * 100
            
            testo_direzione = f"RIALZO tra {orizzonte_giorni} GG 📈" if previsione == 1 else f"RIBASSO tra {orizzonte_giorni} GG 📉"
            
            str.markdown("---")
            str.header(f"🔮 VERDETTO OPERATIVO (PROSSIMI {orizzonte_giorni} GIORNI)")
            mc1, mc2 = str.columns(2)
            mc1.metric(f"Proiezione Direzionale (Swing {orizzonte_giorni} Giorni)", testo_direzione)
            mc2.metric("Confidenza Aggiornata", f"{probabilita:.1f}%")
            
            # --- INVIO AUTOMATICO A TELEGRAM ---
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                messaggio_alert = (
                    f"🤖 *Radar S&P 500 (Swing {orizzonte_giorni} Giorni)*\n\n"
                    f"Direzione Prevista: *{testo_direzione}*\n"
                    f"Confidenza Statistica: *{probabilita:.1f}%*\n\n"
                    f"Sentiment Odierno:\n"
                    f"• Monetario: {punteggi_oggi['Politica Monetaria']:.2f}\n"
                    f"• Geopolitico: {punteggi_oggi['Geopolitica & Crisi']:.2f}"
                )
                invia_messaggio_telegram(messaggio_alert)
                str.success("📲 Segnale Operativo inviato a Telegram!")

else:
    str.warning("Inserisci la NewsAPI Key nella barra laterale per sbloccare il radar.")
