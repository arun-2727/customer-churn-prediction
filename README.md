# 📊 Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn for a telecommunications business, with an interactive Streamlit dashboard, model-wide feature importance, and SHAP-based individual prediction explanations.

## 🚀 Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who may leave can help businesses investigate service issues and develop more effective retention strategies.

This project uses the Telco Customer Churn dataset to build a binary classification pipeline that estimates churn probability from customer demographics, account information, service subscriptions, and billing details.

### Key Features

- **Churn prediction:** Estimate the probability that a customer will churn.
- **Interactive dashboard:** Enter customer information through a Streamlit application.
- **Model interpretability:** Explore global feature importance.
- **Individual explanations:** Use SHAP to investigate features influencing a particular prediction.
- **Model evaluation:** Compare classification models using accuracy, precision, recall, F1-score, and ROC-AUC.
- **Retention insights:** Display illustrative recommendations based on customer characteristics and predicted risk.

## 🖥️ Application

The Streamlit application provides a customer input form, a churn prediction summary, and explainability visualizations.

> Add a screenshot of your working application here.

Save a screenshot in your repository's `assets/` folder and reference it like this:

`![Customer Churn Prediction Dashboard](assets/dashboard.png)`

## 🧠 Machine Learning Workflow

1. **Data inspection and cleaning**
   - Inspect dataset dimensions, data types, missing values, and target distribution.
   - Convert `TotalCharges` to a numeric feature.
   - Handle blank `TotalCharges` values associated with customers with zero tenure.

2. **Exploratory Data Analysis**
   - Analyze churn distribution.
   - Investigate relationships between churn and contract type, tenure, billing, payment method, and subscribed services.
   - Visualize customer and service patterns.

3. **Preprocessing**
   - Separate input features from the churn target.
   - Split the dataset into stratified training and test sets.
   - Impute missing numeric values and scale numeric features.
   - Impute missing categorical values and one-hot encode categorical features.
   - Combine preprocessing and model prediction in a scikit-learn pipeline.

4. **Model training and comparison**
   - Train and compare multiple classification algorithms.
   - Evaluate models using metrics suited to the imbalanced target.
   - Examine the trade-off between identifying churners and generating false-positive predictions.

5. **Explainability**
   - Use global feature importance to inspect influential features.
   - Use SHAP to analyze feature contributions to individual predictions.

6. **Deployment interface**
   - Load the saved model pipeline.
   - Collect customer information through Streamlit.
   - Display estimated churn probability, classification, and illustrative retention recommendations.

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_training_evaluation.ipynb
│
├── models/
│   └── customer_churn_model.pkl
```



## 📊 Dataset

The project uses the Telco Customer Churn dataset.

- **Records:** 7,043 customers
- **Features:** Customer demographics, services, contract details, tenure, and charges
- **Target:** `Churn`
- **Target classes:** `Yes` (churn) and `No` (no churn)

The target distribution is imbalanced: 1,869 customers churned, while 5,174 did not.

## 🔍 Exploratory Data Analysis Findings

The exploratory analysis identified several associations with churn:

- **Contract type:** Month-to-month customers had a churn rate of approximately 42.71%, compared with 11.27% for one-year contracts and 2.83% for two-year contracts.
- **Tenure:** Churned customers had a median tenure of 10 months, compared with 38 months for customers who stayed.
- **Internet service:** Customers using fiber-optic internet had a churn rate of approximately 41.89% in the dataset.
- **Online security and technical support:** Customers without these services showed higher observed churn rates than customers who had them.
- **Payment method:** Electronic-check customers had an observed churn rate of approximately 45.29%.

These are descriptive associations in the dataset; they do not establish that any individual feature causes churn.

## 📈 Model Evaluation

Multiple models were evaluated using a stratified train/test split and classification metrics.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 0.8421 |
| Decision Tree | 79.84% | 63.47% | 56.68% | 59.89% | 0.8276 |
| Random Forest | 76.01% | 53.66% | 70.59% | 60.97% | 0.8343 |
| Gradient Boosting | 79.70% | 65.17% | 50.53% | 56.93% | 0.8453 |
| XGBoost | 80.48% | 67.13% | 51.87% | 58.52% | 0.8467 |
| Weighted XGBoost | 74.73% | 51.56% | 79.68% | 62.61% | 0.8471 |


### Interpreting the results

- **Accuracy** measures the proportion of correctly classified customers.
- **Precision** measures how many predicted churners actually churned.
- **Recall** measures how many actual churners the model identified.
- **F1-score** balances precision and recall.
- **ROC-AUC** measures the model's ability to rank churners above non-churners across thresholds.

Weighted XGBoost achieved higher churn recall in the recorded evaluation, while generating more false positives. This illustrates why model selection should consider business objectives rather than accuracy alone.

## 🔎 Explainability

### Global feature importance

The application can display influential features from the fitted estimator, such as tenure, charges, and contract-related features when supported by the saved model.

### SHAP explanations

SHAP is used to examine the contribution of input features to an individual model prediction.

- Positive SHAP values generally push the explained model output toward the churn class.
- Negative SHAP values generally push it away from the churn class.

The exact interpretation depends on the explainer and output scale. Feature contributions are explanations of model behavior, not evidence of causality.

## 🛠️ Tech Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib
- **Application:** Streamlit
- **Model persistence:** joblib
- **Development:** Jupyter Notebook, Git, GitHub

## ⚙️ Installation and Usage

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure project files

```text
models/customer_churn_model.pkl
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

## ⚠️ Limitations

- Predictions are estimates and are not guarantees of future customer behavior.
- The dataset represents a particular customer population and may not generalize to every telecom business.
- The target is imbalanced, so accuracy alone can be misleading.
- Risk categories and retention suggestions are illustrative and should be validated before business use.
- SHAP explanations describe model behavior and do not establish causal relationships.
- Model performance depends on data quality, preprocessing, evaluation design, and the decision threshold.

## 🔮 Future Improvements

- Deploy a public demo.
- Add model evaluation visualizations, including a confusion matrix and ROC curve.
- Add a batch prediction workflow for multiple customers.
- Validate probability calibration and tune decision thresholds using validation data.
- Improve retention recommendations with business feedback and cost-sensitive evaluation.
- Add automated tests and reproducible training instructions.

## 👨‍💻 Author

**Arun Thota**

Computer Science student specializing in Artificial Intelligence and Machine Learning.

GitHub: [arun-2727](https://github.com/arun-2727)

LinkedIn: [Arun Thota](www.linkedin.com/in/arun-thota-336616331)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
