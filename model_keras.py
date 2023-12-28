import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# 加载数据集
df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
gender_submission_df = pd.read_csv('titanic/gender_submission.csv')

# 数据预处理（与之前相同）
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

# 使用 ColumnTransformer 预处理数据
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = df.drop('Survived', axis=1)
y = df['Survived']
X = preprocessor.fit_transform(X)

# 分割数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建神经网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 预处理测试数据
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
test_df['Age'] = imputer.transform(test_df[['Age']])
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
X_test = test_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
X_test_transformed = preprocessor.transform(X_test)
y_test = gender_submission_df['Survived']


# 进行预测
predictions = model.predict(X_test_transformed)
predictions = [1 if x > 0.5 else 0 for x in predictions]

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test set: {:.2%}".format(accuracy))
