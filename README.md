# 🌫️ Caribbean Air Quality Forecaster

> **A machine learning-based air quality forecasting system developed as a Research Project at the University of the West Indies, Cave Hill Campus.**

This project applies ensemble machine learning models and data augmentation techniques to forecast pollutant levels (CO and O3) across monitoring stations in Trinidad and Tobago. It addresses the challenge of building accurate predictive models in data-scarce Caribbean environments, generating 30-day AQI forecasts with visual output.

---

## Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributors](#-contributors)
- [License](#-license)
- [Contact](#-contact)

---

## About the Project

Air quality forecasting in the Caribbean is particularly challenging due to **sparse monitoring infrastructure** and **limited historical data**. Standard machine learning models trained on small datasets often suffer from overfitting and poor generalisation.

This project (thus far) tackles these issues by:
- Applying **XGBoost**, **Random Forest** and **KNN** models to historical air quality data from Trinidad monitoring stations
- Using **Gaussian Noise Injection** as a data augmentation strategy to improve model robustness
- Generating **30-day recursive forecasts** for CO and O₃ pollutant levels
- Computing **US EPA Air Quality Index (AQI)** values and categories for each forecast day
- Comparing model performance across stations with different environmental profiles

---

## Features

- 30-day recursive CO and O₃ pollutant forecasting
- XGBoost model with time-series cross-validation (no data leakage)
- Gaussian Noise data augmentation for data-scarce environments
- US EPA AQI calculation with colour-coded categories
- Automated chart generation (CO forecast, O₃ forecast, daily AQI bar chart)
- Preprocessing pipeline handling missing values and date formatting

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| XGBoost | Primary forecasting model |
| Scikit-learn | Model evaluation and time-series cross-validation |
| Pandas | Data loading, cleaning and feature engineering |
| NumPy | Numerical operations and noise injection |
| Matplotlib | Forecast visualisations and AQI charts |

---

## Getting Started (ONGOING)

### Prerequisites

- Python 3.9+
- pip
- CSV data files for each monitoring station

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/sharhaynes/Air-Quality-Predictor.git
cd Air-Quality-Predictor
```

**2. Install dependencies**
> ```bash
> pip install pandas numpy matplotlib xgboost scikit-learn
> ```

**3. Add your data files**

Place your CSV files in the root directory:

```
arima_station_data.csv
point_lisas_data.csv
```

Each CSV should contain at minimum the columns: `Date`, `CO`, `O3`

**4. Run the forecaster**

```bash
python predictor2.py
```

This will:
- Train and evaluate the models
- Print a 30-day forecast table to the console
- Save charts to `forecast_charts.png`

---

## Dataset

- Sourced from the **Trinidad and Tobago Environmental Management Authority (EMA)**
- Supplemented by publicly available online repositories where necessary
- Preprocessing steps include removal of NaN values, date parsing and forward/backward filling of O₃ gaps

**Station Notes:**

| Station | CO Data | O₃ Data |
|---|---|---|
| Arima | ✅ Available | ✅ Available |
| Point Lisas | ❌ No sensor | ✅ Available |

> AQI for Point Lisas is calculated from O₃ only due to the absence of a CO sensor.

---

### Training Strategy
- **Time-series split** (5 folds) — model always trains on past data, validates on future data
- **Recursive forecasting** — Day 1 prediction is fed back as input for Day 2, and so on over 30 days

### XGBoost Configuration

| Parameter | Value |
|---|---|
| Estimators | 100 |
| Learning Rate | 0.05 |
| Max Depth | 5 |
| Subsample | 0.8 |

### AQI Calculation
AQI is computed using the **US EPA breakpoint method** for both CO and O₃. The overall daily AQI is the maximum of the two individual pollutant AQI values.

---

## Results

> *(To be updated upon project completion)*

| Metric | Arima | Point Lisas |
|---|---|---|
| CO MAE | TBD | N/A |
| O₃ MAE | TBD | TBD |
| Avg 30-Day AQI | TBD | TBD |

---


---

## Acknowledgements

- Air quality data sourced from the [Trinidad and Tobago Environmental Management Authority (EMA)](https://www.ema.co.tt)
- AQI breakpoint methodology based on [US EPA AQI Technical Assistance Document](https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf)

---

## Contributors

| Name | Role |
|---|---|
| T'Shara Haynes | Developer & Researcher |
| Dr. Thomas Edward| Research Supervisor |

**Institution:** University of the West Indies, Cave Hill Campus
**Department:** Biological and Chemical Sciences
**Course:** ENSC 3020 — Environmental Science Case Study

---

## License

This project is intended for academic purposes. Please contact the author before reuse.

---

## Contact

For questions or inquiries, reach out at: `haynestshara0@gmail.com`
