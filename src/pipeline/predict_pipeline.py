import sys
import pandas as pd
from src.exception import CustomException
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

# CustomData class helps to map the data that we give in the form with the backend.
class CustomData:
    def __init__(self,
                 Pclass: int,
                 Sex: str,
                 Age: float,
                 SibSp: int,
                 Parch: int,
                 Fare: float,
                 Embarked: str,
                 Title: str,
                 FamilySize: int,
                 IsAlone: int,
                 Cabin_Deck: str):
        self.Pclass = Pclass
        self.Sex = Sex
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Fare = Fare
        self.Embarked = Embarked
        self.Title = Title
        self.FamilySize = FamilySize
        self.IsAlone = IsAlone
        self.Cabin_Deck = Cabin_Deck

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Pclass": [self.Pclass], # Given the value in []. pandas.DataFrame expects each column to be a list (or array-like), even if it has only one row.
                "Sex": [self.Sex],
                "Age": [self.Age],
                "SibSp": [self.SibSp],
                "Parch": [self.Parch],
                "Fare": [self.Fare],
                "Embarked": [self.Embarked],
                "Title": [self.Title],
                "FamilySize": [self.FamilySize],
                "IsAlone": [self.IsAlone],
                "Cabin_Deck": [self.Cabin_Deck]
            }

            df = pd.DataFrame(custom_data_input_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)

