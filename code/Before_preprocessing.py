from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neural_network import MLPClassifier as nn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_auc(name, y_pred, color, plot=False):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    if plot == True:
        plt.figure()
        plt.plot(fpr, tpr, color=color, marker ='.', label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic: '+str(name))
        plt.legend(loc="lower right")
        plt.show()

#-------------------------------------------------------------------------------------------------------------------

# Load the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv')

# Show the header and the first five rows
print(df.head())
print(df.shape)

# Retaining only Biopsy as the target variable
df.drop(['Hinselmann', 'Schiller','Citology'], axis=1, inplace=True)
print(df.head())
print(df.shape)


print(df.describe())

df=df.rename(index=str, columns={"Biopsy": "Target"})
print(df.head())

import numpy as np

print('Number of rows before removing rows with missing values: ' + str(df.shape[0]))

# Replace ? with np.NaN
df = df.replace('?', np.NaN)

# Remove rows with np.NaN
df = df.dropna(how='any')

print('Number of rows after removing rows with missing values: ' + str(df.shape[0]))

total = df['Target'].sum()
print('The total number of women having cervical cancer within this 59 observations are :'+  str(total))

# Get the target vector
y = df.Target

# Specify the name of the features
features = list(df.drop('Target', axis=1).columns)

# Get the feature vector
X = df[features]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print([np.where(y_train == 0)[0].shape[0], np.where(y_train == 1)[0].shape[0]])

# # # #-------------------------------------------------------------------------------------------------------------------
# # # Create Decision Tree classifer object

# Train Decision Tree Classifer
clf = dt()
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
predictions = np.array(y_pred)
print(np.unique(predictions))
print("The variance within the predictions are",np.var(predictions))

# Model Accuracy, how often is the classifier correct?
print("Accuracy of the Decision Tree model is:",metrics.accuracy_score(y_test, y_pred))

labels = ["0", '1']
cm = confusion_matrix(y_test, y_pred)
print("Decision Tree",cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier: Decision Tree')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(classification_report(y_test, y_pred))

#PLOT ROC_CURVE
plot_roc_auc('Decision Tree', y_pred, color='blue', plot = True)


# # # #-------------------------------------------------------------------------------------------------------------------

#RandomForestClassifier
clf = rf()
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
predictions = np.array(y_pred)
print(np.unique(predictions))
print("The variance within the predictions are",np.var(predictions))

# Model Accuracy, how often is the classifier correct?
print("Accuracy of the random forest model is:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix

labels = ["0", '1']
cm = confusion_matrix(y_test, y_pred)
print("Random Forest",cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier: Random Forest')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_test, y_pred))

#PLOT ROC_CURVE
plot_roc_auc('Random Forest', y_pred, color='blue', plot = True)

# #-------------------------------------------------------------------------------------------------------------------

#SVM classifier object
clf = SVC(kernel="linear")

# performing training
clf.fit(X_train, y_train)

# make predictions
# predicton on test
y_pred = clf.predict(X_test)
predictions = np.array(y_pred)
print(np.unique(predictions))
print("The variance within the predictions are",np.var(predictions))

labels = ["0", '1']
cm = confusion_matrix(y_test, y_pred)
print("Support Vector Machine",cm)

print("Accuracy of the SVM model is: ", accuracy_score(y_test, y_pred) * 100)
print("\n")

#Confusion matrix plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier: SVM')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# calculate metrics
print("\n")
print("Classification Report: SVM")
print(classification_report(y_test,y_pred))
print("\n")

#PLOT ROC_CURVE
plot_roc_auc('Support Vector Machine', y_pred, color='blue', plot = True)

# # #-------------------------------------------------------------------------------------------------------------------

# standardize the data
# stdsc = StandardScaler()
#
# stdsc.fit(X_train)
#
# X_train_std = stdsc.transform(X_train)
# X_test_std = stdsc.transform(X_test)

#K-Nearest Neighbors
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
predictions = np.array(y_pred)
print(np.unique(predictions))
print("The variance within the predictions are",np.var(predictions))

# Model Accuracy, how often is the classifier correct?
# calculate metrics
print("\n")
print("Classification Report: KNN_7")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy of the knn model is: ", accuracy_score(y_test, y_pred) * 100)
print("\n")
Biopsy_pred = clf.predict(X_test)

labels = ["0", '1']
cm = confusion_matrix(y_test, y_pred)
print("KNN",cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier: KNN_7')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#PLOT ROC_CURVE
plot_roc_auc('KNN_7', y_pred, color='blue', plot = True)

# #-------------------------------------------------------------------------------------------------------------------

# creating the Naive Bayes classifier object
clf = BernoulliNB()
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

# performing training
clf.fit(X_train, y_train)

# make predictions

# predicton on test
y_pred = clf.predict(X_test)
predictions = np.array(y_pred)
print(np.unique(predictions))
print("The variance within the predictions are",np.var(predictions))

y_pred_score = clf.predict_proba(X_test)

print("Accuracy of the Naive Bayes model is : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

# calculate metrics
print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

labels = ["0", '1']
cm = confusion_matrix(y_test, y_pred)
print("Naive Bayes",cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier: Naive Bayes')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#PLOT ROC_CURVE
plot_roc_auc('Naive Bayes', y_pred, color='blue', plot = True)

#-------------------------------------------------------------------------------------------------------------------
