from flask import Flask, render_template, request
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# --- INIZIO NUOVO DATASET (14 Persone, 7 Domande) ---
# REGOLE:
# 1. Genere (0=M, 1=F)
# 2. Occhi (0=Scuri, 1=Chiari)
# 3. Capelli (0=Neri, 1=Marroni, 2=Biondi, 3=Altro)
# 4. Sport (0=Nessuno, 1=Squadra, 2=Individuale)
# 5. Carattere (0=Timido, 1=Estroverso, 2=Altro)
# 6. Orecchini (0=No, 1=Sì)
# 7. Mano (0=Destro, 1=Mancino)
# Etichetta: NOME
dataset = [
    # Vecchi 5
    (0, 0, 0, 0, 0, 0, 0, 'Eyad'),
    (0, 0, 1, 1, 1, 0, 0, 'Cristian'),
    (0, 0, 1, 1, 1, 0, 0, 'Edoardo'),
    (0, 0, 1, 2, 1, 1, 0, 'Emanuele'),
    (0, 0, 1, 1, 1, 0, 0, 'Pietro'),
    # Nuovi 9
    (1, 0, 0, 1, 1, 1, 1, 'Giulia'),
    (0, 0, 0, 1, 1, 0, 0, 'Willson'),
    (0, 0, 1, 1, 1, 0, 0, 'Giovanni'),
    (1, 0, 1, 0, 1, 1, 0, 'Eleonora'),
    (1, 0, 1, 1, 1, 1, 0, 'Maia'),
    (0, 1, 1, 2, 2, 0, 0, 'Riccardo'),
    (0, 0, 0, 2, 0, 0, 0, 'Nicholas'),
    (0, 0, 0, 0, 1, 0, 0, 'Amro'),
    (0, 0, 0, 1, 1, 0, 1, 'Mohamed')
]
# --- FINE DATASET ---

# Addestriamo il modello
X_train = np.array([item[:-1] for item in dataset]) 
y_train = np.array([item[-1] for item in dataset])   
# n_neighbors=3 è un buon numero per 14 campioni
modello = KNeighborsClassifier(n_neighbors=3) 
modello.fit(X_train, y_train)
print(f"✅ Modello 'Indovina Chi' aggiornato con {len(dataset)} persone.")


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    error = None

    if request.method == 'POST':
        try:
            # 1. Prendiamo i valori numerici dalle 7 domande
            val_genere = request.form['genere']
            val_occhi = request.form['occhi']
            val_capelli = request.form['capelli']
            val_sport = request.form['sport']
            val_carattere = request.form['carattere']
            val_orecchini = request.form['orecchini']
            val_mano = request.form['mano']

            # 2. Creiamo la lista di numeri per il modello
            dati_numerici = [
                int(val_genere),
                int(val_occhi),
                int(val_capelli),
                int(val_sport),
                int(val_carattere),
                int(val_orecchini),
                int(val_mano)
            ]

            # 3. Facciamo la predizione
            dati_per_predizione = np.array(dati_numerici).reshape(1, -1)
            predizione = modello.predict(dati_per_predizione)
            prediction_result = predizione[0]

        except Exception as e:
            error = f"Errore durante l'elaborazione: {e}"

    # Mostriamo la pagina HTML
    return render_template('index.html', prediction=prediction_result, error=error)


if __name__ == '__main__':
    app.run(debug=True)