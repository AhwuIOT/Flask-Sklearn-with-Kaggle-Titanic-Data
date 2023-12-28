import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# 加載訓練數據集
df = pd.read_csv('titanic/train.csv')

# 刪除不必要的列
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# 填充缺失值
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 特征工程 - 添加新特征 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch']

# 定義數值型和分類列
numerical_cols = ['Age', 'Fare', 'FamilySize']
categorical_cols = ['Pclass', 'Sex', 'Embarked']

# 創建列轉換器進行數據預處理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# 分離特征和目標變量
X = df.drop('Survived', axis=1)
y = df['Survived']

# 對訓練數據應用轉換
X_train = preprocessor.fit_transform(X)
y_train = y

# 實例化並訓練模型
rad_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rad_model.fit(X_train, y_train)

# 保存模型和預處理管道
pickle.dump(rad_model, open('./model/model2.pkl', 'wb'))
pickle.dump(preprocessor, open('./model/pipeline2.pkl', 'wb'))

# 加載測試數據集和對應的真實標簽
test_df = pd.read_csv('titanic/test.csv')
gender_submission_df = pd.read_csv('titanic/gender_submission.csv')

# 對測試數據集進行同樣的特征工程和預處理
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
test_df['Age'] = imputer.transform(test_df[['Age']])
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
X_test = test_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
X_test_transformed = preprocessor.transform(X_test)
y_test = gender_submission_df['Survived']

# 預測測試數據集
predictions = rad_model.predict(X_test_transformed)

# 計算準確度
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test set: {:.2%}".format(accuracy))

# 創建包含預測結果的 DataFrame
predictions_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})