Titanic Survival Prediction | End-to-End Machine Learning Project
This project predicts whether a passenger survived in the Titanic disaster using machine learning techniques.
It covers the complete end-to-end ML workflow — from data exploration to model deployment.

Project Overview
The Titanic Survival Prediction project aims to build a machine learning model that predicts survival probabilities of passengers aboard the Titanic, based on various features such as age, gender, ticket class, etc.

This project follows a complete ML lifecycle, including:
1. Data Collection and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training and Evaluation
5. Building a Reusable Pipeline
6. Deployment on AWS Elastic Beanstalk

Key Features:
1. End-to-end ML workflow (from raw data to deployment)
2. Automated data preprocessing pipeline
3. Logging and exception handling
4. Modular code structure (reusable components and utilities)
5. Flask-based web application for predictions
6. AWS Beanstalk deployment setup


Tech Stack
Languages & Tools
1. Python 3.13
2. Scikit-learn
3. Pandas, NumPy
4. Matplotlib, Seaborn
5. Flask
6. AWS Elastic Beanstalk

Environment & Version Control
1. Virtual Environment (venv or conda)
2. Git & GitHub
3. VS Code

📊 Workflow
1. Data Exploration (EDA)
Performed in notebook/eda_model_training.ipynb:
Data cleaning and visualization
Handling missing values
Feature correlation analysis
Outlier detection

2. Feature Engineering
Encoding categorical variables
Feature scaling

3. Model Building
Tried multiple models (Logistic Regression, Random Forest, etc.)
Finalized best-performing model based on metrics
Used GridSearchCV for hyperparameter tuning

4. Pipeline Creation
Built using Pipeline from sklearn
Saved preprocessor and model as .pkl files in artifacts/

5. Flask Application
Developed a web app (application.py)
Takes passenger details as input and predicts survival

6. Deployment
Containerized Flask app
Deployed to AWS Elastic Beanstalk

⚙️ Installation and Setup
1. Clone the repository
git clone https://github.com/<your-username>/titanic_survival_prediction.git
cd titanic_survival_prediction

2. Create a virtual environment
python -m venv env
env\Scripts\activate        # For Windows
source env/bin/activate     # For macOS/Linux

3. Install dependencies
pip install -r requirements.txt

4. Run the application locally
python application.py

5. Open the app
Navigate to http://127.0.0.1:5000/ in your browser.

📦 Deployment (AWS Elastic Beanstalk)
Create an AWS account and install the AWS CLI.
Initialize Elastic Beanstalk:
eb init
eb create
eb open
The app will be live on your AWS URL.

🧪 Model Performance
Metric	   Value
Accuracy   	81.56%
Precision	  78.87%
Recall	    75.68%
F1-Score	  77.24%

📁 Artifacts
model.pkl – Trained ML model
preprocessor.pkl – Feature transformation pipeline
train.csv & test.csv – Datasets used for training and testing

🛠️ Logging and Exception Handling
All logs are stored in the logs/ directory.
Custom logger.py and exception.py modules handle runtime events and errors gracefully.

📘 Future Enhancements
Add more models for comparison
Integrate frontend UI improvements
Deploy via Docker + AWS ECS

🤝 Contributing
Contributions are welcome!
Feel free to fork the repository and submit a pull request with improvements.

🧑‍💻 Author
Mounika Maradana
📧 https://www.linkedin.com/in/mounikamaradana/
🌐 https://github.com/Mounika-17

🪪 License
This project is licensed under the MIT License.
