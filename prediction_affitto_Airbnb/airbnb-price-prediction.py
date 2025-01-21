import pandas as pd # importiamo pandas che ci servirà per la manipolazione dei dati
import numpy as np  # importiamo numpy che ci servirà per la manipolazione dei dati
from sklearn.model_selection import train_test_split # importiamo train_test_split per dividere il dataset in training e test set 
from sklearn.linear_model import LinearRegression # importiamo LinearRegression per creare il modello di regressione lineare
from sklearn.preprocessing import StandardScaler # importiamo StandardScaler per standardizzare le feature
from sklearn.metrics import mean_squared_error, r2_score # importiamo mean_squared_error e r2_score per valutare il modello

# Creiamo un dataset di esempio per gli Airbnb
np.random.seed(42) # impostiamo il seed per la riproducibilità
n_samples = 1000 # numero di appartamenti

# Generiamo feature realistiche per gli Airbnb
data = {
    'metri_quadri': np.random.normal(60, 20, n_samples),  # media 60m², deviazione standard 20m²
    'distanza_centro': np.random.uniform(0, 10, n_samples),  # distanza in km dal centro città
    'posti_letto': np.random.randint(1, 7, n_samples),  # da 1 a 6 posti letto
    'bagni': np.random.randint(1, 4, n_samples),  # da 1 a 3 bagni
    'recensioni': np.random.randint(0, 100, n_samples),  # da 0 a 100 recensioni
}

# Creiamo il DataFrame
df = pd.DataFrame(data) # creiamo il DataFrame con i dati generati precedentemente 

# Generiamo i prezzi basati sulle feature (aggiungiamo un po' di rumore per simulare dati reali)
prezzi_base = (
    df['metri_quadri'] * 1.5 +  # €1.5 per metro quadro
    df['posti_letto'] * 25 +    # €25 per posto letto
    df['bagni'] * 30 -          # €30 per bagno
    df['distanza_centro'] * 15 + # -€15 per km dal centro
    df['recensioni'] * 0.1      # €0.1 per recensione
)
df['prezzo'] = prezzi_base + np.random.normal(0, 20, n_samples)  # aggiungiamo rumore

# Separiamo features (X) e target (y)
X = df.drop('prezzo', axis=1) # axis = 1 indica che stiamo eliminando solamente il nome della colonna "prezzo"
y = df['prezzo']              # invece per y teniamo solamente il nome della colonna "prezzo"

# Dividiamo in training e test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizziamo le feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creiamo e addestriamo il modello
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Facciamo predizioni sul test set
y_pred = model.predict(X_test_scaled)

# Valutiamo il modello
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Stampiamo i risultati e i coefficienti
print("Metriche di valutazione:")
print(f"RMSE: {rmse:.2f} €")
print(f"R² Score: {r2:.3f}")
print("\nImportanza delle feature:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

# Esempio di predizione per un nuovo appartamento
nuovo_appartamento = pd.DataFrame({
    'metri_quadri': [75],
    'distanza_centro': [2.5],
    'posti_letto': [4],
    'bagni': [2],
    'recensioni': [50]
})

# Standardizziamo le feature del nuovo appartamento
nuovo_appartamento_scaled = scaler.transform(nuovo_appartamento)

# Prediciamo il prezzo
prezzo_predetto = model.predict(nuovo_appartamento_scaled)[0]
print(f"\nPrezzo predetto per il nuovo appartamento: {prezzo_predetto:.2f} €")
# stampiamo il prezzo del nuovo appartamento
