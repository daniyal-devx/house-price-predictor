# House Price Predictor API

A machine learning API that predicts house sale prices in Pakistan, built with XGBoost, FastAPI, and Docker. Trained on real estate listing data scraped from Zameen.com covering five major Pakistani cities.

## Live Demo

- **API base URL:** _add your Render URL here after deployment_
- **Interactive docs (Swagger UI):** `<your-render-url>/docs`

## Overview

This project takes raw property listing data and turns it into a deployable prediction service. It covers the full pipeline: data cleaning, feature engineering, model comparison, model serving via a REST API, and containerized cloud deployment.

## Tech Stack

- **Language:** Python 3.13
- **ML:** scikit-learn, XGBoost
- **API:** FastAPI, Pydantic, Uvicorn
- **Deployment:** Docker, Render
- **Data handling:** pandas, numpy

## Dataset

- **Source:** [Pakistan House Price dataset](https://www.kaggle.com/datasets/jillanisofttech/pakistan-house-price-dataset) (Zameen.com listings), via Kaggle
- **Size:** 168,446 raw listings → 120,652 after cleaning
- **Coverage:** Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad
- **Known limitation:** listings span Aug 2018 – Jul 2019. Absolute price levels are dataset-era prices, not current market prices. The model is best used to compare *relative* pricing between properties, not as a live valuation tool.

## Data Cleaning & Feature Engineering

Key steps taken (see `notebooks/01_eda.ipynb` for the full walkthrough):

1. **Filtered to "For Sale" listings only.** The raw data mixed sale and rental listings under one `price` column — combining them would corrupt the target variable, since rent and sale price are entirely different scales.
2. **Fixed a mixed-unit bug.** Property size was recorded in two incompatible units (Marla and Kanal, where 1 Kanal = 20 Marla) inside the same numeric column. All sizes were converted to a single `area_marla` unit before use.
3. **Removed genuine data errors, not just statistical outliers.** Rather than blanket-clipping extreme values, each extreme price/area was manually inspected. Legitimate high-end listings (e.g. Gulberg, Clifton mansions) were kept; a physically impossible listing (a "Lower Portion" recorded at 12,000 Marla) was removed as a clear entry error.
4. **Dropped non-predictive columns:** listing IDs, page URLs, agency/agent (high cardinality, no real price signal, ~26% missing), and date fields (insufficient date range to model seasonality).
5. **Dropped `location` (neighborhood name).** At 1,455 unique values, this was too high-cardinality for safe encoding within this project's scope. `latitude`/`longitude` were kept as a numeric proxy for location instead. Target-encoding `location` is a noted future improvement (see below).
6. **Log-transformed the target (`price`).** Raw price was heavily right-skewed (a small number of very expensive properties dominated the distribution). Log-transforming produced a near-normal distribution that both models could fit properly.

## Model Development & Results

Three modeling stages were compared to justify the final model choice:

| Model | Target | R² | MAE (PKR) | Notes |
|---|---|---|---|---|
| Linear Regression | raw price | 0.167 | ~16.9M | Poor fit; skewed target breaks linear assumptions |
| Linear Regression | log(price) | 0.337 | unstable / unusable | Better fit, but extrapolated to nonsensical predictions on extreme inputs (a single outlier area value produced a multi-trillion PKR prediction) |
| **XGBoost** | **log(price)** | **0.885** | **~4.82M** | Final model. Tree-based splits stay within the range of training data, avoiding the extrapolation failure seen above |

**Why XGBoost was chosen over Linear Regression:** beyond the R² improvement, Linear Regression showed a concrete failure mode — when given feature combinations far outside the training distribution (e.g. a very large property), its unbounded linear extrapolation produced predictions in the trillions of PKR after inverse log-transforming. XGBoost, as a tree-based model, cannot predict outside the bounds of values it saw during training, making it far more robust for a public-facing API where input values can't be fully controlled.

### Feature Importance (XGBoost)

| Feature | Importance |
|---|---|
| `area_marla` | 52.4% |
| `property_type_House` | 10.1% |
| `bedrooms` | 5.6% |
| `city_Rawalpindi` | 5.0% |
| `latitude` | 4.3% |
| `longitude` | 2.3% |

Property size dominates the prediction, consistent with real-world real estate pricing intuition. `latitude`/`longitude` carry a modest signal as a substitute for the dropped `location` feature — this gap is the clearest lever for future improvement.

## API Usage

### Health check
```
GET /
```
```json
{"status": "ok", "message": "House Price Predictor API is running"}
```

### Predict
```
POST /predict
Content-Type: application/json
```

**Request body:**
```json
{
  "property_type": "House",
  "city": "Lahore",
  "province_name": "Punjab",
  "latitude": 31.5204,
  "longitude": 74.3587,
  "baths": 3,
  "bedrooms": 3,
  "area_marla": 8.0
}
```

**Response:**
```json
{
  "predicted_price_pkr": 16454096
}
```

If `city` or `property_type` isn't one of the values seen during training, the response includes a `warnings` field flagging that the prediction may be unreliable, rather than failing silently.

## Running Locally

```bash
git clone https://github.com/daniyal-devx/house-price-predictor.git
cd house-price-predictor
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
uvicorn src.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive API docs.

## Running with Docker

```bash
docker build -t house-price-predictor .
docker run -p 8000:8000 house-price-predictor
```

## Project Structure

```
house-price-predictor/
├── data/              # raw dataset (gitignored)
├── notebooks/
│   └── 01_eda.ipynb   # full data cleaning + modeling walkthrough
├── src/
│   └── main.py        # FastAPI app
├── models/
│   ├── house_price_model.pkl
│   └── model_columns.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

## Future Improvements

- Target-encode `location` (with a proper train-only encoding to avoid leakage) to recover the neighborhood-level price signal currently lost
- Hyperparameter tuning (current XGBoost run uses untuned defaults)
- Refresh with more recent listing data to correct for dataset age
- Add prediction confidence intervals rather than a single point estimate

## Author

**Daniyal Usman**
- GitHub: [daniyal-devx](https://github.com/daniyal-devx)
- LinkedIn: [daniyalusman-dev](https://linkedin.com/in/daniyalusman-dev)
