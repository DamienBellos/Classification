import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

knn = pd.read_csv('KNN_Project_Data')
print(knn.head())

# Pairplot to graphically compare data categories
sns.pairplot(knn, hue='TARGET CLASS', palette='RdBu')
plt.show()

# scale the objects
scaler = StandardScaler()
scaler.fit(knn.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(knn.drop('TARGET CLASS', axis=1))

# Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.
knn_feat = pd.DataFrame(scaled_features, columns=knn.columns[:-1])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, knn['TARGET CLASS'], test_size=0.3)

# Create a KNN model instance with n_neighbors=1
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
knn_model = knn_model.predict(X_test)

# Predictions and Evaluations
print(classification_report(knn_model, y_test))
print('\n')
print(confusion_matrix(y_test, knn_model))

# Choose a K Value using the elbow method to pick a good K Value!
error_rate = []
for i in range(1, 40):
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train, y_train)
    pred_i = knn_i.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

#  Plot using the information from the for loop.
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='--', marker="o",
         markerfacecolor='Red', markersize='10')
plt.title('Error Rate vs K Value')
plt.ylabel('Error Rate')
plt.xlabel('K')
plt.show()

# Retrain with the optimized K Value
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('With n_neighbors=15')
print('\n')
print(classification_report(pred, y_test))
print('\n')
print(confusion_matrix(pred, y_test))
