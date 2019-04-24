
# coding: utf-8

# <h1 align="center"> 
#  BIOPSY CLASSIFICATION OF CERVICAL CANCER
# </h1>
# 
# <h4 align="center"> 
# 
# </h4>

# 
# <h3 align="center"> 
#  Authors: Sam Aboagye,Swetha Kalla and Armandi Heydarian
# <h3 align="center">
# Date: 04/23/2019
# </h3>

# ## DATA: 
# The dataset was collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela. The dataset comprises demographic information, habits, and historic medical records of 858 patients. Several patients decided not to answer some of the questions because of privacy concerns (missing values).
# 
# Source:
# Kelwin Fernandes (kafc _at_ inesctec _dot_ pt) - INESC TEC & FEUP, Porto, Portugal. 
# Jaime S. Cardoso - INESC TEC & FEUP, Porto, Portugal. 
# Jessica Fernandes - Universidad Central de Venezuela, Caracas, Venezuela.
# 
# ### ATTRIBUTES: 
# (int) Age 
# (int) Number of sexual partners 
# (int) First sexual intercourse (age) 
# (int) Num of pregnancies 
# (bool) Smokes 
# (bool) Smokes (years) 
# (bool) Smokes (packs/year) 
# (bool) Hormonal Contraceptives 
# (int) Hormonal Contraceptives (years) 
# (bool) IUD 
# (int) IUD (years) 
# (bool) STDs 
# (int) STDs (number) 
# (bool) STDs:condylomatosis 
# (bool) STDs:cervical condylomatosis 
# (bool) STDs:vaginal condylomatosis 
# (bool) STDs:vulvo-perineal condylomatosis 
# (bool) STDs:syphilis 
# (bool) STDs:pelvic inflammatory disease 
# (bool) STDs:genital herpes 
# (bool) STDs:molluscum contagiosum 
# (bool) STDs:AIDS 
# (bool) STDs:HIV 
# (bool) STDs:Hepatitis B 
# (bool) STDs:HPV 
# (int) STDs: Number of diagnosis 
# (int) STDs: Time since first diagnosis 
# (int) STDs: Time since last diagnosis 
# (bool) Dx:Cancer 
# (bool) Dx:CIN 
# (bool) Dx:HPV 
# (bool) Dx 
# (bool) Hinselmann: target variable 
# (bool) Schiller: target variable 
# (bool) Cytology: target variable 
# (bool) Biopsy: target variable
# 
# ### OVERVIEW:
# This project aims to evaluate and compare different classiferson the Cervical Cancer dataset. There are 4 targets, but for the purposes of this project only one of them which is the Biopsy will be classified. The target classes are boolean but have been encoded. Any target with a value of 1 is encoded for True which represents Malignant cervical cancer and any target with a value of 0 represent represents benign cervical cancer.
# 
# Classifiers:
# * Logistic Regression
# * Random Forest
# * Support Vector Machine
# * Neural Network 
# 

# ### Import Dependencies

# In[43]:





import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore")


# ### Load the Cervical Cancer data

# In[44]:



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv')


df.head()


# In[45]:


#Descriptive statistics 
df.describe()


# ### Check the Cervical Cancer Information

# In[46]:


#df.info()


# ## 3. Data preprocessing

# #3.1 Drop these columns
# #('STDs: Time since first diagnosis','STDs: Time since last diagnosis', 
# #which are targets with significant missing values)
# #Hinselmann,Schiller,Citology;Additional Targets not needed)
# #3.2 Rename Biopsy Column as Target
# #3.3 Impute Missing Values

# In[47]:


df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis','Hinselmann','Schiller','Citology'],inplace=True,axis=1)


# In[48]:


df=df.rename(index=str, columns={"Biopsy": "target"})
df.head()


# In[49]:



print('Number of rows before removing rows with missing values: ' + str(df.shape[0]))

# Replace ? with np.NaN
df = df.replace('?', np.NaN)

# Remove rows with np.NaN
df = df.dropna(how='any')

print('Number of rows after removing rows with missing values: ' + str(df.shape[0]))
df.head()


# ### Get the Count of the classes and plot them

# In[50]:


print(pd.DataFrame(df.target.value_counts()))
sns.set(style="darkgrid")
ax = sns.countplot(x="target",hue="target",data= df,palette="Set1")


# In[51]:


#X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values


# In[52]:


#Get the target vector
y = df.target
# Specify the name of the features
features = list(df.drop('target', axis=1).columns)

# Get the feature vector
X = df[features]


# In[53]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print([np.where(y_train == 0)[0].shape[0], np.where(y_train == 1)[0].shape[0]])


# In[60]:


#Scale the data
std_scl= StandardScaler()

X_train_std = std_scl.fit_transform(X_train)

X_test_std = std_scl.transform(X_test)


# In[61]:


# Apply PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print("Explained Variance:\n")
print(pca.explained_variance_ratio_)


# In[68]:


plt.plot(pca.explained_variance_ratio_, alpha=0.5)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()


# In[77]:


# Plot the principal components

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


# In[62]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_sample(X_train, y_train)

print([np.where(y_train == 0)[0].shape[0], np.where(y_train == 1)[0].shape[0]])


# In[63]:


# Apply PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print("Explained Variance:\n")
print(pca.explained_variance_ratio_)


# In[75]:


clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test) 
# Print the classification accuracy
print('The accuracy of random forest is: ' + str(clf.score(X_test, y_test)))
print("\n")
print(np.unique(y_predict))
print("Classification Report: ")
print(classification_report(y_test,y_predict))
print("\n")
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_predict, labels=[0, 1]))


# In[73]:



clf = SVC()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test) 
# Print the classification accuracy
print('The accuracy of SVC is: ' + str(clf.score(X_test, y_test)))
print("\n")
print(np.unique(y_predict))
print("Classification Report: ")
print(classification_report(y_test,y_predict))
print("\n")
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predict, labels=[0, 1])


# In[76]:


clf = LogisticRegression()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test) 
# Print the classification accuracy
print('The accuracy of Logistic Regression  is: ' + str(clf.score(X_test, y_test)))
print("\n")
print(np.unique(y_predict))
print("Classification Report: ")
print(classification_report(y_test,y_predict))
print("\n")
confusion_matrix(y_test, y_predict, labels=[0, 1])


# In[67]:


# Apply PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print("Explained Variance:\n")
print(pca.explained_variance_ratio_)


# In[78]:


from sklearn.metrics import precision_recall_fscore_support

def train_test_evaluate(classifier):
    """
    Train, test, and evaluate the classifier
    :param classifier: a classifier             
    """
    
    # Declare the model
    clf = classifier(random_state=0)
    
    # Train the model
    clf.fit(X_train, y_train)
    
    if classifier is DecisionTreeClassifier:
        global tree  
        # Get the tree
        tree = clf
    elif classifier is RandomForestClassifier:
        global importances
        # Get the feature importances
        importances = clf.feature_importances_
    
    # Update the list of accuracies
    accuracies.append(clf.score(X_test, y_test))


# In[79]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# The list of classifiers
clfs = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

# The tree
tree = None

# The list of importances
importances = []

# The list of accuracies
accuracies = []

# For each classifer
for classifier in clfs:
    # Call function train_test_evaluate (defined above)
    train_test_evaluate(classifier)


# In[80]:


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from IPython.display import Image

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['0', 
                                        '1'],
                           feature_names=X.columns,
                           out_file=None) 

graph = graph_from_dot_data(dot_data) 

img_cancer = Image(graph.create_png()) 
img_cancer


# In[81]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[82]:


clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels

f_importances = pd.Series(importances, df.iloc[:, 1:].columns)



# sort the array in descending order of the importances

f_importances.sort_values(ascending=False, inplace=True)



# make the bar Plot from f_importances

f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)



# show the plot

plt.tight_layout()

plt.show()


# #These features were the most important for predictive accuracy of the model
# #1.Number of Sexual partners 
# #2.IUD 
# #3.Num of Pregnancies 
# #4'Smokes 
# #5.First Sexual Intercourse 
# #6.Hormonal Contraceptives 
# #7.STDS 
# #8.Smokes(packs/year)
# #9.Hormonal contraceptives 
# #10.Dx:CIN(Abnormal cells are found on the surface of the cervix.
#        CIN is usually caused by certain types of human papillomavirus (HPV) and is 
#        found when a cervical biopsy is done.)
