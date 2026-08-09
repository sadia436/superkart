import io
import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_api = Flask("SuperKart Sales Prediction API")
model = joblib.load("superkart_model.joblib")
FEATURE_COLUMNS = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP", "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char", "Store_Age_Years", "Product_Type_Category"]

@superkart_api.get("/")
def home():
    return jsonify({"message": "Welcome to the SuperKart Sales Prediction API"})

@superkart_api.get("/health")
def health():
    return jsonify({"status": "healthy"})

def validate_features(frame):
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return frame[FEATURE_COLUMNS]

@superkart_api.post("/v1/predict")
def predict():
    try:
        input_data = validate_features(pd.DataFrame([request.get_json(force=True)]))
        return jsonify({"predicted_sales": round(float(model.predict(input_data)[0]), 2)})
    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Upload a CSV file using the field name file."}), 400
        input_data = pd.read_csv(io.BytesIO(request.files["file"].read()))
        input_data["predicted_sales"] = model.predict(validate_features(input_data)).round(2)
        return jsonify(input_data.to_dict(orient="records"))
    except (ValueError, TypeError, KeyError, pd.errors.EmptyDataError) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=True)
