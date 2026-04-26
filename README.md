# Credit Card Fraud Detection

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AA4A44?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)

A machine learning project that tackles credit card fraud detection as a binary classification problem on an imbalanced dataset, comparing multiple ML models across different sampling strategies to find the most effective approach.

---

##  Problem Statement

Credit card fraud is a major financial threat where fraudulent transactions are extremely rare compared to legitimate ones. This class imbalance makes it difficult for standard classifiers to detect fraud reliably. The goal of this project is to build and compare ML models that can accurately identify fraudulent transactions while minimising false negatives, since missing a fraud case is costlier than a false alarm.

---

##  Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)
- **Size:** 14,446 credit card transactions
- **Class Distribution:** Highly imbalanced, fraudulent transactions are a small minority
- **Key Features:** `amt`, `category`, `job`, `city`, `lat`, `long`, `city_pop`, `dob`, `trans_date_trans_time`, `is_fraud`

---

##  Methodology

### 1. Exploratory Data Analysis
- Plotted distributions of key features against the target variable to understand patterns and class separation

### 2. Feature Engineering
- Extracted `trans_year`, `trans_month`, `trans_day` from transaction timestamps
- Derived customer `Age` from date of birth and transaction year
- Dropped irrelevant or redundant columns after feature extraction: `merch_lat`, `merch_long`, `trans_num`, `merchant`, `state`, `dob`, `trans_date_trans_time`

### 3. Preprocessing
- Encoded categorical columns:: `category`, `job`, `city`
- Scaled numerical columns to a uniform range: `amt`, `lat`, `long`, `city_pop`, `Age`, `trans_year`, `trans_month`, `trans_day`

### 4. Handling Class Imbalance
Three strategies were tested independently:
- **No Sampling** — Trained on original imbalanced data
- **Undersampling** — Reduced majority class to match minority
- **Oversampling** — Increased minority class to match majority

### 5. Models Trained
All four models were trained and evaluated under each sampling strategy using 5-fold cross-validation:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

##  Results

Performance on the Fraud class (Precision, Recall, F1-Score):

| Model | Sampling | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|
| Decision Tree | No Sampling | 99.08 | 99.26 | 99.17 |
| Random Forest | No Sampling | 99.82 | 99.63 | 99.72 |
| XGBoost | No Sampling | 99.63 | 99.45 | 99.54 |
| Decision Tree | Undersampling | 87.68 | 98.16 | 92.63 |
| Random Forest | Undersampling | 92.97 | 99.63 | 96.18 |
| XGBoost | Undersampling | 96.78 | 99.45 | 98.10 |
| Decision Tree | Oversampling | 99.45 | 99.63 | 99.54 |
| Random Forest | Oversampling | 99.45 | 99.45 | 99.45 |
| XGBoost | Oversampling | 99.63 | 100.00 | 99.82 |

 **Best Model:** XGBoost with Oversampling achieved 100% Recall and 99.82% F1-Score on the fraud class.

---

##  Key Findings

- **XGBoost with oversampling** achieved the best overall performance with perfect recall, meaning it caught every fraudulent transaction in the test set.
- **Random Forest without sampling** performed surprisingly well on imbalanced data, suggesting ensemble methods handle class skew better than single-tree approaches.
- **Undersampling consistently degraded precision**, particularly for Decision Tree, indicating that discarding majority class samples loses useful decision boundary information.
- **Oversampling generalised better** than undersampling across all three models, making it the recommended strategy for this type of imbalanced fraud dataset.


---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/BalajiRamGanesh/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies
```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn
```

### 3. Run the notebook
Open `credit_card_fraud_detection.ipynb` in Jupyter Notebook and run all cells sequentially.

---

