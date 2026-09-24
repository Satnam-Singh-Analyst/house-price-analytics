# 🏡 Real Estate Price Intelligence & Predictive Analytics

## 📌 Project Overview
This project applies Machine Learning and Business Intelligence to real-estate data. The Streamlit application cleans housing data, performs Exploratory Data Analysis (EDA), trains a **Random Forest Regressor** to forecast property prices, and turns the results into an executive risk-and-action summary.

**Key features**
- KPI dashboard (average price, square footage, inventory size, property age)
- Data cleaning (type coercion, missing/duplicate/invalid row removal)
- EDA: price distribution, correlation with price, summary statistics
- Model evaluation: R², 5-fold cross-validated R², RMSE, MAE, feature importance, actual-vs-predicted plot
- Interactive price prediction with an 80% uncertainty range
- Executive Risk & Action Matrix computed directly from the data
- Optional CSV upload so the same pipeline runs on real data

## 🔗 Dataset Information
* **Reference dataset:** Kaggle *House Prices – Advanced Regression Techniques*
* **Source link:** [https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
* **Data used by default:** a reproducible **synthetic dataset** (500 properties, fixed random seed) that mirrors the core Kaggle schema (`SquareFeet`, `Bedrooms`, `Bathrooms`, `Age`, `Price`), so the app runs with no downloads.
* **Using real data:** upload any CSV containing the columns `SquareFeet, Bedrooms, Bathrooms, Age, Price` through the sidebar (e.g. a Kaggle extract mapped to these names: `GrLivArea→SquareFeet`, `BedroomAbvGr→Bedrooms`, `FullBath→Bathrooms`, `YrSold−YearBuilt→Age`, `SalePrice→Price`).

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd IBM-Bob-Data-Analytics-Project
   ```
2. Install dependencies (Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## 📊 Key Insights & Business Recommendations
*(Figures below are from the default synthetic dataset; the app recalculates them live for any uploaded data.)*
- **Primary driver:** Square footage dominates price (≈98% of feature importance, correlation ≈ 0.99).
- **Model accuracy:** R² ≈ 0.99 on held-out data, MAE ≈ $13.6k. These are high because the synthetic data follows a near-linear formula; expect lower scores on real-world data.
- **Risk:** Properties older than 30 years sell for ≈2.9% less per sq ft than properties aged ≤10 years; budget for repair liabilities.
- **Recommended action:** Prioritise acquisition of larger (>2,500 sqft), newer (<10 years) properties and apply a repair buffer to older stock.

## 📁 Repository Structure
```
├── app.py            # Data processing, ML model, Streamlit UI
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── Project_Report.docx
```

## ⚠️ Disclaimer
Predictions are statistical estimates and are not a substitute for a professional property appraisal.
