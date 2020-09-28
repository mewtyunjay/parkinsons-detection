from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('bgcl_classification_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction_proba = model.predict_proba(final_features)
        
        output = prediction_proba[0][1]

        if(output>0.7):
            return render_template('index.html', prediction_text="The patient is in danger. The chances of Parkinson's Disease in the patient are {}".format(output))
        else:
            return render_template('index.html', prediction_text="The patient is safe. The chances of Parkinson's Disease in the patient are very low.")


if __name__=="__main__":
    app.run(debug=True)

