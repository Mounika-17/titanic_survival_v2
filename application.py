from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)
app=application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# When we visit the URL directly in a browser → the GET block runs. The default HTTP method is GET
# When we submit the form → it sends a POST request → the POST block runs.

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Extract data from form
        data = CustomData(
            Pclass = int(request.form.get('Pclass')),
            Sex = request.form.get('Sex'),
            Age = float(request.form.get('Age')),
            SibSp = int(request.form.get('SibSp')),
            Parch = int(request.form.get('Parch')),
            Fare = float(request.form.get('Fare')),
            Embarked = request.form.get('Embarked'),
            Title = request.form.get('Title'),
            FamilySize = int(request.form.get('FamilySize')),
            IsAlone = int(request.form.get('IsAlone')),
            Cabin_Deck = request.form.get('Cabin_Deck')
        )
        # Covert the above data into dataframe
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        # Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=int(results[0]))

       
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)