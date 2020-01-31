import pandas as pd
import pickle
import json

from flask import Flask, request

application = Flask("python_data_analysis")

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@application.route("/predict", methods=["GET"])
def predict():
    content = request.json
    target = pd.DataFrame.from_dict(content)
    print(target)
    pred = model.predict(target)
    
    res = []
    for i in range(len(target)):
        doc = {}
        doc['row'] = i
        doc['predicted'] = pred[i]
        res.append(doc)
    return json.dumps(res)

if __name__ ==  "__main__":
    application.run(host="0.0.0.0", port=80)