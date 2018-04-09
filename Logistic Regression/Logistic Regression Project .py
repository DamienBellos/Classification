
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set_style('whitegrid')

ad_data = pd.read_csv('advertising.csv')
ad_data.head(8)
ad_data.info()

# Plots a histogram of the age
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
sns.distplot(ad_data['Age'], bins=30, kde=False, color='blue')
plt.show()

# Jointplot showing Area Income versus Age.
sns.jointplot(x='Age', y='Area Income', data=ad_data)
plt.show()

# Jointplot showing the kde distributions of daily time spent on site vs age.
ad_data.columns
sns.distplot(np.random.random(100))
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde', color='r')
plt.show()

# Jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='g')
plt.show()

# Pairplot with the hue defined by the 'Clicked on Ad'
sns.pairplot(ad_data, hue='Clicked on Ad', palette='RdBu_r')
plt.show()

# Logistic Regression
ad_data.columns

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train and fit a logistic regression model on the training set.
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Predict and Evaluat
predict = logmodel.predict(X_test)
print(classification_report(y_test, predict), confusion_matrix(y_test, predict))
