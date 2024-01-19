
from flask import Flask, request
from flask_restful import Api, Resource
import pandas as pd
import pickle

app = Flask(__name__)
api = Api(app)

# Load pipeline and model using the binary files
model = pickle.load(open('./model/rad_model.pkl', 'rb'))
pipeline = pickle.load(open('./model/pipeline.pkl', 'rb'))

# Function to test if the request contains multiple 
def islist(obj):
  return True if ("list" in str(type(obj))) else False

class Preds(Resource):
  def put(self):
    json_ = request.json
    # If there are multiple records to be predicted, directly convert the request json file into a pandas dataframe
    if islist(json_['PassengerId']):
      entry = pd.DataFrame(json_)
    # In the case of a single record to be predicted, convert json request data into a list and then to a pandas dataframe
    else:
      entry = pd.DataFrame([json_])
    # Transform request data record/s using the pipeline
    entry_transformed = pipeline.transform(entry)
    # Make predictions using transformed data
    prediction = model.predict(entry_transformed)
    res = {'predictions': {}}
    # Create the response
    for i in range(len(prediction)):
      res['predictions'][i + 1] = int(prediction[i])
    return res, 200 # Send the response object

api.add_resource(Preds, '/predict')

if __name__ == "__main__":
  app.run(debug = True)