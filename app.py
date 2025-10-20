from flask import Flask, render_template, request
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# --- INIZIO DATASET "SUPER SPECIFICO" ---
# REGOLE:
# 1. Genere (0=M, 1=F)
# 2. Occhi (0=Scuri, 1=Chiari)
# 3. Capelli (0=Neri, 1=Marroni)
# 4. Sport (0=Nessuno, 1=Calcio, 2=Scacchi, 3=Basket, 4=Pallavolo, 5=Kendo, 6=MMA, 7=Football)
# 5. Carattere (0=Timido, 1=Determinato, 2=Amichevole, 3=Estroverso, 4=Sportivo, 5=Solare/divertente, 
#             6=Tendente estrov., 7=Solare, 8=Gentile, 9=Brutto, 10=Introverso/silenz., 11=Solare/sociev., 12=Ottimista)
# 6. Orecchini (0=No, 1=Sì)
# 7. Mano (0=Destro, 1=Mancino)
# Etichetta: NOME
dataset = [
    #Gen,Och,Cap,Spo,Car,Ore,Man, NOME
    (0, 0, 0, 0, 0,  0, 0, 'Eyad'),
    (0, 0, 1, 1, 1,  0, 0, 'Cristian'),
    (0, 0, 1, 1, 2,  0, 0, 'Edoardo'),
    (0, 0, 1, 2, 3,  1, 0, 'Emanuele'),
    (0, 0, 1, 3, 4,  0, 0, 'Pietro'),
    (1, 0, 0, 4, 5,  1, 1, 'Giulia'),
    (0, 0, 0, 3, 6,  0, 0, 'Willson'),
    (0, 0, 1, 3, 7,  0, 0, 'Giovanni'),
    (1, 0, 1, 0, 8,  1, 0, 'Eleonora'),
    (1, 0, 1, 4, 7,  1, 0, 'Maia'),
    (0, 1, 1, 5, 9,  0, 0, 'Riccardo'),
    (0, 0, 0, 6, 10, 0, 0, 'Nicholas'), # <-- Eccolo! Con Sport=6 e Carattere=10
    (0, 0, 0, 0, 11, 0, 0, 'Amro'),
    (0, 0, 0, 7, 12, 0, 1, 'Mohamed')
]
# --- FINE DATASET ---

# Addestriamo il modello KNN (che ora è il nostro "fallback")
X_train = np.array([item[:-1] for item in dataset]) 
y_train = np.array([item[-1] for item in dataset])   
modello = KNeighborsClassifier(n_neighbors=3) 
modello.fit(X_train, y_train)
print(f"✅ Modello 'IBRIDO SUPER SPECIFICO' aggiornato con {len(dataset)} persone.")


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

            # --- LA LOGICA IBRIDA IF/ELSE ---
            
            # 3. (IF) Controlliamo prima se esiste una corrispondenza ESATTA
            prediction_result = None 
            dati_da_cercare = tuple(dati_numerici)
            
            for item in dataset:
                caratteristiche_dataset = item[:-1] 
                if caratteristiche_dataset == dati_da_cercare:
                    prediction_result = item[-1] # Trovato!
                    break 

            # 4. (ELSE) Se non abbiamo trovato una corrispondenza esatta...
            if prediction_result is None:
                # ...allora usiamo il metodo dei vicini (KNN)
                print("Nessuna corrispondenza esatta, uso KNN...") 
                
                dati_per_predizione = np.array(dati_numerici).reshape(1, -1)
                predizione_knn = modello.predict(dati_per_predizione)
                prediction_result = predizione_knn[0] + " (Ipotesi)" 
            
            # --- FINE DELLA LOGICA IF/ELSE ---

        except Exception as e:
            error = f"Errore durante l'elaborazione: {e}"

    # Mostriamo la pagina HTML
    return render_template('index.html', prediction=prediction_result, error=error)


if __name__ == '__main__':
    app.run(debug=True)