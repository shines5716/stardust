# =======================
# 1. IMPORTS
# =======================
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# =======================
# 2. CREATE FLASK APP
# =======================
app = Flask(__name__)

# =======================
# 3. LOAD TRAINED MODEL
# (RF.pkl must be in the SAME folder as test.py)
# =======================
with open("RF.pkl", "rb") as file:
    model = pickle.load(file)

# =======================
# 4. HOME ROUTE
# =======================
@app.route("/")
def home():
    return render_template("home.html")

# =======================
# 5. SUBMIT / PREDICT ROUTE
# =======================
@app.route("/submit", methods=["POST"])
def submit():
    try:
        # Read input values from form
        input_feature = [float(x) for x in request.form.values()]

        # Feature names (MUST match training order)
        names = [
            'i', 'z', 'modelFlux_z', 'petroRad_g', 'petroRad_r',
            'petroFlux_z', 'petroR50_u', 'petroR50_g',
            'petroR50_i', 'petroR50_r'
        ]

        # Convert to DataFrame
        data = pd.DataFrame([input_feature], columns=names)

        # Predict
        prediction = model.predict(data)

        # Map output
        if prediction[0] == 0:
            result = "starforming"
        else:
            result = "starbursting"

        return render_template("output.html", prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

# =======================
# 6. RUN SERVER
# =======================
if __name__ == "__main__":
    app.run(debug=True, port=2222)
