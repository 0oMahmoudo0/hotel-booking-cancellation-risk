# Hotel Booking Cancellation Risk

> **One-sentence problem:** Predict, at the time of booking, whether a hotel reservation will be cancelled — so revenue managers can apply the right deposit policy before money is lost.

---

## Live App

🔗 **[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)**  *(replace with your actual URL after deployment)*

---

## Screenshot

> *Add a screenshot here after first deployment.*
> `![App screenshot](screenshot.png)`

---

## Results

### Cross-Validation (training set)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Baseline (majority class) | 0.500 | 0.275 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.818 | 0.623 | 0.793 | 0.497 | 0.611 |
| Random Forest | 0.867 | 0.715 | 0.503 | 0.698 | 0.584 |
| HistGradientBoosting | 0.861 | 0.701 | 0.488 | 0.682 | 0.569 |
| XGBoost | 0.865 | 0.709 | 0.517 | 0.679 | 0.587 |

### Hold-out Test Set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (majority class) | 0.725 | 0.000 | 0.000 | 0.000 | 0.500 | 0.275 |
| Logistic Regression | 0.719 | 0.493 | 0.785 | 0.606 | 0.815 | 0.615 |
| Random Forest (tuned) | 0.806 | 0.695 | 0.522 | 0.596 | 0.870 | 0.719 |
| HistGradientBoosting (tuned) | 0.803 | 0.684 | 0.525 | 0.594 | 0.868 | 0.712 |
| **Soft-Voting Ensemble (RF + HGB + XGB)** | **0.807** | **0.695** | **0.531** | **0.602** | **0.872** | **0.722** |
| XGBoost (tuned) | 0.803 | 0.678 | 0.543 | 0.603 | 0.869 | 0.716 |

> Numbers are taken directly from notebook output. No invented metrics.

---

## How to Run Locally

### Prerequisites

- Python **3.12 or 3.13** (the model pickle requires this)
- The three model files in the same directory as `app.py`:
  - `app_model.joblib`
  - `app_calibrator.joblib`
  - `app_metadata.json`

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

### Run the automated tests

```bash
python -m pytest test_app.py -v
```

---

## How to Deploy on Streamlit Community Cloud

1. Push this repository (including `app_model.joblib`, `app_calibrator.joblib`, `app_metadata.json`) to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Advanced settings** → set **Python version** to `3.12` or `3.13`.
6. Click **Deploy**.

> The model files are ~4 MB total, so no Git LFS is needed.

---

## Limitations

- **Temporal scope:** Trained on data from two hotels in Portugal, 2015–2017. Performance may differ for other properties, countries, or future time periods.
- **Booking-time only:** The model uses only information available when the booking is made. Post-booking signals (check-in actions, room upgrades) are not included.
- **Accuracy drift:** Accuracy on future bookings is typically lower than on historical test data.
- **Thresholds:** Risk thresholds were derived from assumed revenue-loss cost ratios (`missed_cancel : false_alarm = 1 : 0.25`). They may not be optimal for every property's pricing strategy.
- **Not causal:** A high score means the booking *looks similar to* past cancellations — it does not prove the guest *will* cancel.

---

## Dataset Credit

Antonio, N., de Almeida, A., & Nunes, L. (2019). **Hotel booking demand datasets**. *Data in Brief*, 22, 41–49.  
Available on Kaggle: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

> The raw dataset is **not** included in this repository.

---

## Repository Structure

```
.
├── app.py                    # Streamlit application
├── app_model.joblib          # Trained scikit-learn pipeline (~4 MB)
├── app_calibrator.joblib     # Isotonic calibrator
├── app_metadata.json         # Thresholds, column lists, categories
├── requirements.txt          # Pinned dependencies
├── test_app.py               # Automated AppTest suite
└── README.md
```

