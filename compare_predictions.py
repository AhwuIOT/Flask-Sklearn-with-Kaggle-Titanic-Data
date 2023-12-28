# 與真實存活比較

import pandas as pd

# Load the two CSV files
predictions_df = pd.read_csv('./output/predictions.csv')
comparison_df = pd.read_csv('./titanic/gender_submission.csv')
predictions_rad_df = pd.read_csv('./output/predictions_rad.csv')


# Compare the 'Survived' column in both dataframes
differences = predictions_df[predictions_df['Survived'] != comparison_df['Survived']]
differences_rad = predictions_rad_df[predictions_rad_df['Survived'] != comparison_df['Survived']]
accuracy = 1 - (len(differences) / len(comparison_df))
accuracy_rad = 1 - (len(differences_rad) / len(comparison_df))
print("Accuracy of predictions: {:.2%}".format(accuracy))
print("Accuracy of predictions_rad: {:.2%}".format(accuracy_rad))
# Output the differences
print("Number of different predictions:", len(differences))
if len(differences) > 0:
    print("\nDifferences found in predictions:")
    print(differences)
    differences.to_csv('./output/differences.csv', index=False)

if len(differences_rad) > 0:
    print("\nDifferences found in predictions_rad:")
    print(differences_rad)
    differences_rad.to_csv('./output/differences_rad.csv', index=False)
