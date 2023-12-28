from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# 加載數據集
df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
gender_submission_df = pd.read_csv('titanic/gender_submission.csv')

# 特征工程 - 添加新特征 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch']
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

# 定義數值型和分類列
numerical_cols = ['Age', 'Fare', 'FamilySize']
categorical_cols = ['Pclass', 'Sex', 'Embarked']

# 創建數值型數據的處理 Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# 創建分類數據的處理 Pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

# 創建 ColumnTransformer 整合所有處理步驟
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# 刪除不必要的列
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# 分離特征和目標變量
X = df.drop('Survived', axis=1)
y = df['Survived']

# 應用 ColumnTransformer
X_train = preprocessor.fit_transform(X)
y_train = y

# 實例化並訓練模型
rad_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rad_model.fit(X_train, y_train)

# 保存模型和預處理管道
pickle.dump(rad_model, open('./model/model2.pkl', 'wb'))
pickle.dump(preprocessor, open('./model/pipeline2.pkl', 'wb'))

# 保存 PassengerId 以便後續使用
passenger_ids = test_df['PassengerId'].copy()

# 測試數據集預處理
test_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
X_test = preprocessor.transform(test_df)
y_test = gender_submission_df['Survived']

# 預測測試數據集
predictions = rad_model.predict(X_test)

# 計算準確度
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test set: {:.2%}".format(accuracy))

# 創建包含預測結果的 DataFrame
predictions_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
