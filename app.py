from flask import Flask, request, jsonify, render_template  # type: ignore
import joblib  # type: ignore
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignorer les avertissements de versions incompatibles
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Charger les modèles et leurs performances
try:
    models = joblib.load("model.pkl", mmap_mode='r')
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du fichier model.pkl : {e}")

# Sélectionner un modèle par défaut (par exemple, RandomForest)
selected_model_key = "RandomForest"  # Remplacez par un autre modèle si nécessaire
if selected_model_key not in models:
    raise KeyError(f"Le modèle '{selected_model_key}' n'existe pas dans model.pkl.")
selected_model = models[selected_model_key]["model"]

# Initialiser l'application Flask
app = Flask(__name__)

@app.route("/")
def index():
    # Afficher la page d'accueil avec le formulaire
    return render_template("index.html", prediction=None, probability=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les features envoyées depuis le formulaire
        features = [float(request.form[f"feature{i}"]) for i in range(1, 14)]  # Supposons 13 features
        
        # Effectuer la prédiction avec le modèle par défaut
        prediction = selected_model.predict([features])[0]
        probability = selected_model.predict_proba([features])[0][1] * 100  # Probabilité de churn (%)
        
        # Retourner la page avec les résultats
        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability, 2),
        )
    except ValueError:
        return jsonify({"error": "Veuillez entrer des valeurs numériques pour toutes les features."}), 400
    except Exception as e:
        return jsonify({"error": f"Une erreur est survenue : {str(e)}"}), 400

if __name__ == "__main__":
    app.run(debug=True)
