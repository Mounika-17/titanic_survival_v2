import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# cv_scores.mean() → Average accuracy across folds. Tells you the central tendency (how well the model performs on average).
# cv_scores.std() → Standard deviation across folds. Tells you how stable/consistent the model’s performance is across different folds.
# y_pred_test → final predicted labels on test set.
# y_proba_test → predicted probabilities for the positive class (useful for ROC-AUC, precision-recall curves, threshold tuning).


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # Composite scoring formula
    def composite_score(self, acc, f1, roc_auc, cv_std):
        # Higher acc, f1, and roc_auc are better; lower std is better
        # You can tune these weights based on your goal (balance, stability, etc.)
        score = (0.4 * acc) + (0.4 * f1) + (0.2 * (roc_auc if roc_auc else 0))
        # Optionally penalize unstable models
        score -= (cv_std * 0.05)
        return score


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train/test arrays")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Dictionary of models and param grids
            models_and_params = {
                "LogisticRegression": (
                    LogisticRegression(max_iter=5000),
                    {
                        'C': [0.01, 0.1, 1, 10],
                        'penalty': ['l2'],
                        'fit_intercept': [True, False]
                    }
                ),
                "DecisionTree": (
                    DecisionTreeClassifier(),
                    {
                        'max_depth': [3, 5, 10, None],
                        'min_samples_split': [2, 5, 10]
                    }
                ),
                "RandomForest": (
                    RandomForestClassifier(),
                    {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5, 10]
                    }
                ),
                "GradientBoosting": (
                    GradientBoostingClassifier(),
                    {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5]
                    }
                ),
                "SVC": (
                    SVC(probability=True),
                    {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf']
                    }
                ),
                "KNN": (
                    KNeighborsClassifier(),
                    {
                        'n_neighbors': [3, 5, 7, 9]
                    }
                ),
            
                "XGBoost": ( 
                   XGBClassifier(eval_metric="logloss"),
                   {
                       'n_estimators': [100, 200],
                       'learning_rate': [0.01, 0.1, 0.2],
                       'max_depth': [3, 5, 7]
                   }
                )
            }

            best_overall_model = None
            best_overall_score = -1
            best_overall_results = {}

            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # Loop through models
            for model_name, (model, param_grid) in models_and_params.items():
                logging.info(f"Running GridSearchCV for {model_name}")

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")

                logging.info(f"{model_name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                # Test evaluation
                y_pred_test = best_model.predict(X_test)
                if hasattr(best_model, "predict_proba"):
                    y_proba_test = best_model.predict_proba(X_test)[:, 1]
                else:
                    # fallback for models without predict_proba (like some SVC)
                    y_proba_test = None

                acc = accuracy_score(y_test, y_pred_test)
                prec = precision_score(y_test, y_pred_test)
                rec = recall_score(y_test, y_pred_test)
                f1 = f1_score(y_test, y_pred_test)
                roc = roc_auc_score(y_test, y_proba_test) if y_proba_test is not None else None

                logging.info(f"{model_name} Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
                             f"F1: {f1:.4f}, ROC-AUC: {roc if roc else 'N/A'}")
                score = self.composite_score(acc, f1, roc, cv_scores.std())
                if score > best_overall_score:
                    best_overall_model = best_model
                    best_overall_score = score
                    best_overall_results = {
                        "model": model_name,
                        "best_params": grid_search.best_params_,
                        "cv_accuracy_mean": cv_scores.mean(),
                        "cv_accuracy_std": cv_scores.std(),
                        "test_accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "roc_auc": roc,
                        "composite_score": score
                    }

            # if acc > best_overall_score:
            #     best_overall_model = best_model
            #     best_overall_score = acc
            #     best_overall_results = {
            #         "model": model_name,
            #         "best_params": grid_search.best_params_,
            #         "cv_accuracy_mean": cv_scores.mean(),
            #         "cv_accuracy_std": cv_scores.std(),
            #         "test_accuracy": acc,
            #         "precision": prec,
            #         "recall": rec,
            #         "f1": f1,
            #         "roc_auc": roc
            #     }

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_overall_model
            )

            logging.info(f"Best overall model: {best_overall_results['model']} "
                         f"with Test Accuracy {best_overall_results['test_accuracy']:.4f}")

            return best_overall_results

        except Exception as e:
            raise CustomException(e, sys)
