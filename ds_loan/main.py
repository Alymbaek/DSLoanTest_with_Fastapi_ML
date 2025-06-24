import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

ds_loan_app = FastAPI(title='Predict DSLoanTest')


class DSLoanTestSchema(BaseModel):
    loan_id: int
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int



BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / 'model_ds.pkl'
scaler_path = BASE_DIR / 'scaler_ds.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


@ds_loan_app.post('/predict/')
async def predict_ds_loan(ds: DSLoanTestSchema):
    ds_dict = ds.model_dump()
    education_own = ds_dict.pop('education')
    education_1_or_0 = [
        1 if education_own == 'Not Graduate' else 0,
    ]
    self_employed_own = ds_dict.pop('self_employed')
    self_employed_1_or_0 = [
        1 if education_own == 'Yes' else 0,
    ]

    features = list(ds_dict.values()) + education_1_or_0 + self_employed_1_or_0

    scaled_features = scaler.transform([features])
    pred = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]

    return {'approved': pred, 'prob': round(prob,2)}



if __name__ == '__main__':
    uvicorn.run(ds_loan_app, host='127.0.0.1', port=8001)





