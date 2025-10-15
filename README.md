# 🆕 Note:
In this version of the Titanic Survival Prediction project, the preprocessing and model training steps are integrated into a **single serialized pipeline (`model.pkl`)**, eliminating the need for a separate `preprocessor.pkl` file.  

#  Titanic Survival Prediction | End-to-End Machine Learning Project

This project predicts whether a passenger survived in the Titanic disaster using machine learning techniques.  
It covers the complete **end-to-end ML workflow** — from data exploration to model deployment.

---

##  Project Overview

The **Titanic Survival Prediction** project aims to build a machine learning model that predicts survival probabilities of passengers aboard the Titanic, based on various features such as age, gender, ticket class, etc.

This project follows a **complete ML lifecycle**, including:

1. Data Collection and Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Model Training and Evaluation  
5. Building a Reusable Pipeline  
6. Deployment on AWS Elastic Beanstalk  

---

## ✨ Key Features

1. End-to-end ML workflow (from raw data to deployment)  
2. Automated data preprocessing pipeline  
3. Logging and exception handling  
4. Modular code structure (reusable components and utilities)  
5. Flask-based web application for predictions  
6. AWS Beanstalk deployment setup  

---

## 🧰 Tech Stack

### Languages & Tools
1. Python 3.13  
2. Scikit-learn  
3. Pandas, NumPy  
4. Matplotlib, Seaborn  
5. Flask  
6. AWS Elastic Beanstalk  

### Environment & Version Control
1. Virtual Environment (`venv` or `conda`)  
2. Git & GitHub  
3. VS Code  

---

## 📊 Workflow

### 1. Data Exploration (EDA)
Performed in `notebook/eda_model_training.ipynb`:  
- Data cleaning and visualization  
- Handling missing values  
- Feature correlation analysis  
- Outlier detection  

### 2. Feature Engineering
- Encoding categorical variables  
- Feature scaling  

### 3. Model Building
- Tried multiple models (Logistic Regression, Random Forest, etc.)  
- Finalized best-performing model based on metrics  
- Used `GridSearchCV` for hyperparameter tuning  

### 4. Pipeline Creation
- Built using `Pipeline` from `sklearn`  
- In this version, preprocessing and model training are combined into a **single pipeline**  
- The complete pipeline is saved as `model.pkl` in the `artifacts/` directory

### ⚙️ Why Preprocessing and Model Training Are Combined in a Single Pipeline

In this version of the project, the **preprocessing** and **model training** steps are integrated into a single pipeline and saved together as `model.pkl`.  
This design was chosen to ensure **proper cross-validation** and **avoid data leakage** during model selection.

#### 🔍 Explanation

When performing model selection using `GridSearchCV`, the data is internally split into multiple folds for cross-validation.  
If preprocessing (such as encoding, scaling, or imputation) is done **before** cross-validation, the transformations are fitted on the entire dataset — including data from validation folds.  
This leads to **data leakage**, causing the model to see information from the validation set during training, which artificially boosts performance metrics.

To prevent this, the pipeline is constructed as follows:

```python
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

Then, GridSearchCV is applied on the full pipeline, ensuring that: 

- For each CV split, preprocessing is fitted only on the training fold 
- The corresponding validation fold remains unseen during transformation and training 
- All model hyperparameters are searched correctly without leakage 

✅ Benefits 
- Prevents data leakage in internal CV 
- Ensures fair and reproducible evaluation metrics 
- Simplifies deployment — the single model.pkl file contains both preprocessing and model steps 

### 5. Flask Application
- Developed a web app (`application.py`)  
- Takes passenger details as input and predicts survival  

### 6. Deployment
- Containerized Flask app  
- Deployed to AWS Elastic Beanstalk   

## ⚙️ Installation and Setup  
### 1. Clone the repository  
git clone https://github.com/Mounika-17/titanic-survival-prediction-ml.git  
cd titanic_survival_prediction  

### 2. Create a virtual environment  
python -m venv env  
env\Scripts\activate        # For Windows  
source env/bin/activate     # For macOS/Linux  

### 3. Install dependencies  
pip install -r requirements.txt  

### 4. Run the application locally  
python application.py  

### 5. Open the app  
Navigate to http://127.0.0.1:5000/ in your browser.  

## 📦 Deployment (AWS Elastic Beanstalk)  
Create an AWS account and install the AWS CLI.  
Initialize Elastic Beanstalk:  
eb init  
eb create  
eb open  
The app will be live on your AWS URL.  


## 🧪 Model Performance

I evaluated multiple machine learning models to predict the target outcome. The models were assessed using cross-validation accuracy and test set metrics including precision, recall, F1 score, and ROC-AUC. Among all models, XGBoost performed the best, achieving the highest test accuracy and balanced overall performance across all metrics.

| Model              | CV Accuracy (mean ± std) | Test Accuracy | Precision  | Recall     | F1 Score   | ROC-AUC    |
| ------------------ | ------------------------ | ------------- | ---------- | ---------- | ---------- | ---------- |
| LogisticRegression | 0.7894 ± 0.0233          | 0.7821        | 0.7333     | 0.7432     | 0.7383     | 0.8425     |
| DecisionTree       | 0.7753 ± 0.0258          | 0.7821        | 0.7160     | 0.7838     | 0.7484     | 0.8328     |
| RandomForest       | 0.7950 ± 0.0321          | 0.7989        | 0.7714     | 0.7297     | 0.7500     | 0.8591     |
| GradientBoosting   | 0.8034 ± 0.0351          | 0.8101        | 0.7703     | 0.7703     | 0.7703     | 0.8674     |
| SVC                | 0.7922 ± 0.0255          | 0.7877        | 0.7500     | 0.7297     | 0.7397     | 0.8256     |
| KNN                | 0.7781 ± 0.0222          | 0.7765        | 0.7576     | 0.6757     | 0.7143     | 0.8429     |
| **XGBoost**        | **0.8034 ± 0.0362**      | **0.8212**    | **0.7838** | **0.7838** | **0.7838** | **0.8678** |

**🏆 Best Model:** XGBoost, achieving the highest Test Accuracy **(0.8212)**, with balanced precision, recall, and F1 score., making it the final selected model for deployment.


## 📁 Artifacts
model.pkl – Combined preprocessing and trained ML pipeline  
train.csv & test.csv – Datasets used for training and testing 

## 🛠️ Logging and Exception Handling  
All logs are stored in the logs/ directory.  
Custom logger.py and exception.py modules handle runtime events and errors gracefully.  

## 📘 Future Enhancements  
Add more models for comparison  
Integrate frontend UI improvements  
Deploy via Docker + AWS ECS  

## 🤝 Contributing  
Contributions are welcome!  
Feel free to fork the repository and submit a pull request with improvements.  

## 🧑‍💻 Author  
Mounika Maradana  
📧 https://www.linkedin.com/in/mounikamaradana/  
🌐 https://github.com/Mounika-17  

## 🪪 License  
This project is licensed under the MIT License.  
