# 將test.csv作預測驗證
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
# Load the trained model and pipeline
model_path = './model/model.pkl'
pipeline_path = './model/pipeline.pkl'
rad_model_path = './model/rad_model.pkl'

with open(model_path, 'rb') as model_file, open(pipeline_path, 'rb') as pipeline_file\
      , open(rad_model_path, 'rb')as rad_model_file:
    trained_model = pickle.load(model_file)
    full_pipeline = pickle.load(pipeline_file)
    train_rad_model = pickle.load(rad_model_file)


# Load the test dataset
test_df = pd.read_csv('./titanic/test.csv')
gender_submission_df = pd.read_csv('titanic/gender_submission.csv')
# Preprocess the test dataset
X_test = test_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
X_test_transformed = full_pipeline.transform(X_test)
y_test = gender_submission_df['Survived']
# Predict survival on the test dataset
predictions = trained_model.predict(X_test_transformed)
predictions_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
predictions_rad = train_rad_model.predict(X_test_transformed)
predictions_rad_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions_rad})
# Save predictions to a CSV file
predictions_df.to_csv('./output/predictions.csv', index=False)
predictions_rad_df.to_csv('./output/predictions_rad.csv', index=False)
# accuracy = accuracy_score(y_test, predictions_rad)
# print("Accuracy on test set: {:.2%}".format(accuracy))