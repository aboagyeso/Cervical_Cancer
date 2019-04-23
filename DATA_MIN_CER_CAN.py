
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

# In[181]:





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

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore")


# ### Load the Cervical Cancer data

# In[182]:



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv')


df.head()


# In[183]:


#Descriptive statistics 
df.describe()


# ### Check the Cervical Cancer Information

# In[184]:


#df.info()


# ## 3. Data preprocessing

# #3.1 Drop these columns
# #('STDs: Time since first diagnosis','STDs: Time since last diagnosis', 
# #which are targets with significant missing values)
# #Hinselmann,Schiller,Citology;Additional Targets not needed)
# #3.2 Rename Biopsy Column as Target
# #3.3 Impute Missing Values

# In[185]:


df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis','Hinselmann','Schiller','Citology'],inplace=True,axis=1)


# In[186]:


df=df.rename(index=str, columns={"Biopsy": "Target"})
df.head()


# #### replace missing values with np.mean

# In[187]:


df = df.replace('?', np.NaN)
#df = df.fillna(df.mean())
df = df.dropna(how='any')
df.head()


# #### Check the total number of missing values in each column

# In[188]:


df.isnull().sum()


# ### Get the Count of the classes and plot them

# In[189]:


pd.DataFrame(df.Target.value_counts())


# In[190]:


sns.set(style="darkgrid")
ax = sns.countplot(x="Target",hue="Target",data= df,palette="Set1")


# In[191]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print([np.where(y_train == 0)[0].shape[0], np.where(y_train == 1)[0].shape[0]])


# In[ ]:


#RANDOMFORESTCLASSIFIER


# In[195]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_predict = rf.predict(X_test) 
# Print the classification accuracy
print('The accuracy of random forest is: ' + str(rf.score(X_test, y_test)))
print(f1_score(y_test, y_predict))
print(recall_score(y_test, y_predict))


# In[193]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred, labels=[0, 1])


# In[45]:


# Class count
count_class_0, count_class_1 = df.Target.value_counts()

# Divide by class
df_class_0 = df[df['Target'] == 0]
df_class_1 = df[df['Target'] == 1]


# In[46]:


#Address class Imbalance With Over Sampling


# In[47]:


#RANDOM OVERSAMPLING
#Since our data set is small we choose to Oversample to compensate for imbalance classes


# In[48]:


df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.Target.value_counts())
sns.set(style="darkgrid")
ax = sns.countplot(x="Target",hue="Target",data= df_test_over,palette="Set1")


# In[49]:


# Get the feature vector
X = df.drop('Target', axis=1).values

# Get the target vector
y = df['Target'].values


# ### 3.3. Get the feature and target vector

# In[50]:


# Specify the name of the target
target = 'Target'

# Get the target vector
y = df[target]

# Specify the name of the features
features = list(df.drop(target, axis=1).columns)

# Get the feature vector
X = df[features]


# ### 3.5. Divide the data into training and testing

# In[126]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print([np.where(y_train == 0)[0].shape[0], np.where(y_train == 1)[0].shape[0]])


# In[124]:



# Separate majority and minority classes
df_majority = df[df.Target==0]
df_minority = df[df.Target==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=623,    # to match majority class
                                 random_state=123) # reproducible results
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# Display new class counts
df_upsampled.Target.value_counts()


# In[125]:



# Separate input features (X) and target variable (y)
y = df_upsampled.Target
X = df_upsampled.drop('Target', axis=1)
 
# Train model
clf_1 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_1 = clf_1.predict(X)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
 
# How's our accuracy?
print( accuracy_score(y, pred_y_1) )


# In[120]:



from sklearn.utils import resample
# Separate input features and target
y = df.Target
X = df.drop('Target', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)




X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_cancer = X[X.Target==0]
cancer = X[X.Target==1]

# upsample minority
cancer_upsampled = resample(cancer,
                          replace=True, # sample with replacement
                          n_samples=len(not_cancer), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_cancer, cancer_upsampled])

# check new class counts
upsampled.Target.value_counts()


# In[121]:


# trying logistic regression again with the balanced dataset
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score
y_train = upsampled.Target
X_train = upsampled.drop('Target', axis=1)
upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)

# Checking accuracy
print(accuracy_score(y_test, upsampled_pred))
    
    
# f1 score
print(f1_score(y_test, upsampled_pred))
    
    
print(recall_score(y_test, upsampled_pred))
#print(classification_score(y_test, upsampled_pred)))


# In[122]:


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict()


# ### 3.6.Scale the Data

# In[89]:



std_scl= StandardScaler()

X_train_std = std_scl.fit_transform(X_train)

X_test_std = std_scl.transform(X_test)


# In[90]:


#FIT THE RANDOM FOREST CLASSIFIER AND GET THE ACCURACY


# In[91]:


from sklearn.ensemble import RandomForestClassifier
# Delcare the model
rf = RandomForestClassifier(random_state=0, class_weight='balanced')
# Train the model
rf.fit(X_train,y_train)
# Print the classification accuracy
print('The accuracy of random forest is: ' + str(rf.score(X_test, y_test)))


# In[92]:


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


# In[93]:


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


# In[196]:


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


# In[197]:


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


# clf = RandomForestClassifier(n_estimators=100)
# 
# # perform training
# clf.fit(X_train, y_train)
# importances = clf.feature_importances_
# 
# # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
# 
# f_importances = pd.Series(importances, df.iloc[:, 1:].columns)
# 
# 
# 
# # sort the array in descending order of the importances
# 
# f_importances.sort_values(ascending=False, inplace=True)
# 
# 
# 
# # make the bar Plot from f_importances
# 
# f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
# 
# 
# 
# # show the plot
# 
# plt.tight_layout()
# 
# plt.show()

# In[74]:


#GET THE FEATURE IMPORTANCE OF ALL THE FEATURES AND CREAT A DICTIONARY


# In[75]:


# Convert the importances into one-dimensional 1d array with corresponding df column names as axis labels
f_importances = f_importances = pd.Series(rf.feature_importances_, features)
# Sort the array in descending order of the importances
f_importances = f_importances.sort_values(ascending=False)


# #Draw the bar plot of feature importance

# In[76]:


f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16,9), rot=90, fontsize=16,color='red')

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

# In[77]:


feature_Descriptions=(['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
       'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
       'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
       'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
       'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
       'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
       'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Target'])


# In[78]:


# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.025
# Train the selector
# Print the names of the most important features


# In[79]:


from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(rf, threshold=0.025)
important_features =[]  
sfm.fit(X_train, y_train)
for feature_list_index in sfm.get_support(indices=True):
    important_features.append(features[feature_list_index])
#print(predictions, row.names = FALSE)
pd.DataFrame(important_features)


# In[80]:


y_pred =rf.predict(X_test)


# In[81]:


#Get unique values in the prediction
print(np.unique(y_pred))


# In[ ]:


from sklearn.utils import resample


# In[ ]:


confmat_lrcv = confusion_matrix(y_true = y_test, y_pred=y_pred_lrcv)
conf_plot(confmat_lrcv)
print('F1: %.3f' % f1_score(y_true=y_test,y_pred=y_pred_lrcv))


# ### 3.7.Dimensionality Reduction Using PCA

# ### The fit learns some quantities from the data, most importantly the "components" and "explained variance":

# In[ ]:


pca = PCA()

X_train_pca = pca.fit_transform(X_train_std)
                   
Explained_Variance = (pca.explained_variance_ratio_)
print("The PCA Components:")


print(pca.components_)


# In[ ]:


print("Explained Variance:")

print(Explained_Variance)


# In[ ]:


# Plot the principal components
X_train_pca = pca.fit_transform(X_train_std)
plt.plot(np.cumsum(Explained_Variance))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()


# In[ ]:



# Apply PCA



pca = PCA()

X_train_pca = pca.fit_transform(X_train_std)



print("Explained Variance:\n")

print(pca.explained_variance_ratio_)


# In[ ]:


###first 20 components contain approximately 90% of the variance,
##while you need around 30 components to describe close to 100% of the variance.


# In[ ]:


plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])

plt.xlabel('PC 1')

plt.ylabel('PC 2')

plt.show()


# #USE THE FEATURE ELIMINATION FUNCTION FROM SKLEARN

# In[ ]:



from sklearn.svm import SVC
#from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load the digits dataset
#digits = load_digits()
X = df.reshape
y = df.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Delcare the model
rf = RandomForestClassifier(random_state=0, class_weight='balanced')
# Train the model
rf.fit(X_train,y_train)
# Print the classification accuracy
print('The accuracy of random forest is: ' + str(rf.score(X_test, y_test)))


# In[ ]:



from sklearn.feature_selection import RFE
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
selector.support_ 
array([ True,  True,  True,  True,  True, False, False, False, False,
       False])
selector.ranking_
array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])


# ### Poting the Explained Variance

# In[ ]:


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)

X_test_pca = pca.transform(X_test_std)


# In[ ]:


lr = LogisticRegression()

lr = lr.fit(X_train_pca, y_train)


# In[ ]:



# Plot decision regions on train data



pp.plot_decision_regions(X_train_pca, y_train, classifier=lr)

plt.xlabel('PC 1')

plt.ylabel('PC 2')

plt.legend(loc='lower left')

plt.tight_layout()

plt.show()


# In[ ]:


# Apply LDA



lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train_std, y_train)


# In[ ]:



# Apply logistic regression after LDA



lr = LogisticRegression()

lr = lr.fit(X_train_lda, y_train)



pp.plot_decision_regions(X_train_lda, y_train, classifier=lr)

plt.xlabel('LD 1')

plt.ylabel('LD 2')

plt.legend(loc='lower left')

plt.tight_layout()

plt.show()


# In[ ]:


X_test_lda = lda.transform(X_test_std)



pp.plot_decision_regions(X_test_lda, y_test, classifier=lr)

plt.xlabel('LD 1')

plt.ylabel('LD 2')

plt.legend(loc='lower left')

plt.tight_layout()

plt.show()


# In[ ]:


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
selector.support_ 


selector.ranking_


# ### 3.6. Over sampling

# In[ ]:


get_ipython().system('pip install imbalanced-learn')
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_sample(X_train, y_train)

print([np.where(y_train == 0)[0].shape[0], np.where(y_train == 1)[0].shape[0]])


# In[ ]:


#perform training with random forest with all columns

# specify random forest classifier

clf = RandomForestClassifier(n_estimators=100)



# perform training

clf.fit(X_train, y_train)


# In[ ]:


#%%-----------------------------------------------------------------------

#plot feature importances

# get feature importances

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


# In[ ]:


#%%-----------------------------------------------------------------------

#select features to perform training with random forest with k columns

# select the training dataset on k-features

newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:15]]



# select the testing dataset on k-features

newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:15]]


# In[ ]:


#%%-----------------------------------------------------------------------

#perform training with random forest with k columns

# specify random forest classifier

clf_k_features = RandomForestClassifier(n_estimators=100)



# train the model

clf_k_features.fit(newX_train, y_train)


# In[ ]:


#%%-----------------------------------------------------------------------

#make predictions



# predicton on test using all features

y_pred = clf.predict(X_test)

y_pred_score = clf.predict_proba(X_test)



# prediction on test using k features

y_pred_k_features = clf_k_features.predict(newX_test)

y_pred_k_features_score = clf_k_features.predict_proba(newX_test)


# In[ ]:


# %%-----------------------------------------------------------------------

# calculate metrics gini model



print("\n")

print("Results Using All Features: \n")



print("Classification Report: ")

print(classification_report(y_test,y_pred))

print("\n")



print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

print("\n")



print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)



# calculate metrics entropy model

print("\n")

print("Results Using K features: \n")

print("Classification Report: ")

print(classification_report(y_test,y_pred_k_features))

print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)

print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)


# In[ ]:


# %%-----------------------------------------------------------------------

# confusion matrix for gini model

conf_matrix = confusion_matrix(y_test, y_pred)

class_names = data['diagnosis'].unique()





df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )



plt.figure(figsize=(5,5))



hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)



hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)

hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)

plt.ylabel('True label',fontsize=20)

plt.xlabel('Predicted label',fontsize=20)

# Show heat map

plt.tight_layout()


# In[ ]:


# %%-----------------------------------------------------------------------



# confusion matrix for entropy model



conf_matrix = confusion_matrix(y_test, y_pred_k_features)

class_names = data['diagnosis'].unique()





df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )



plt.figure(figsize=(5,5))



hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)



hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)

hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)

plt.ylabel('True label',fontsize=20)

plt.xlabel('Predicted label',fontsize=20)

# Show heat map

plt.tight_layout()

plt.show()


# In[ ]:


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


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.

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
    
print(importances)


# In[ ]:


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

Image(graph.create_png()) 


# In[ ]:


import matplotlib.pyplot as plt

# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X.columns)

# Sort the array in descending order of the importances
f_importances = f_importances.sort_values(ascending=False)

# Draw the bar Plot from f_importances 
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(12,10), rot=45, fontsize=10)

# Show the plot
plt.tight_layout()
cervi = plt.savefig('feat_importance')
plt.show()


# ## 4. Hypterparameter tuning and model selection
# In this section, we will first use the combination of pipeline and GridSearchCV to tune the hyperparameters of five classifiers:
# - logistic regression
# - multi-layer perceptron
# - decision tree
# - random forest
# - support vector machine
# 
# Next we will select the best model across the five classifiers.

# ### 4.1. Create the dictionary of classifiers
# In the dictionary:
# - the key is the acronym of the classifier
# - the value is the classifier (with random_state=0)

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

clfs = {'lr': LogisticRegression(random_state=0),
        'mlp': MLPClassifier(random_state=0),
        'dt': DecisionTreeClassifier(random_state=0),
        'rf': RandomForestClassifier(random_state=0),
        'svc': SVC(random_state=0, probability=True)}

# clfs = {'lr': LogisticRegression(random_state=0, class_weight='balanced'),
#         'mlp': MLPClassifier(random_state=0),
#         'dt': DecisionTreeClassifier(random_state=0, class_weight='balanced'),
#         'rf': RandomForestClassifier(random_state=0, class_weight='balanced'),
#         'svc': SVC(random_state=0, probability=True, class_weight='balanced')}


# ### 4.2. Create the dictionary of pipeline
# In the dictionary:
# - the key is the acronym of the classifier
# - the value is the pipeline (with StandardScaler and the classifier)

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe_clfs = {}

for name, clf in clfs.items():
    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()), ('clf', clf)])


# ### 4.3. Create the dictionary of parameter grids
# In the dictionary:
# - the key is the acronym of the classifier
# - the value is the parameter grid of the classifier

# In[ ]:


param_grids = {}


# ### 4.3.1. The parameter grid for logistic regression
# The hyperparameters we want to tune are:
# - multi_class
# - solver
# - C
# 
# Here we need to use two dictionaries in the parameter grid since 'multinomial' (multi_class) does not support 'liblinear' (solver). See details of the meaning of the hyperparametes in [sklearn logistic regression documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# In[ ]:


C_range = [10 ** i for i in range(-4, 5)]

param_grid = [{'clf__multi_class': ['ovr'], 
               'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'clf__C': C_range},
              {'clf__multi_class': ['multinomial'],
               'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
               'clf__C': C_range}]

param_grids['lr'] = param_grid


# ### 4.3.2. The parameter grid for multi-layer perceptron
# The hyperparameters we want to tune are:
# - hidden_layer_sizes
# - activation
# 
# See details of the meaning of the hyperparametes in [sklearn multi-layer perceptron documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)

# In[ ]:


param_grid = [{'clf__hidden_layer_sizes': [10, 100, 200],
               'clf__activation': ['identity', 'logistic', 'tanh', 'relu']}]

param_grids['mlp'] = param_grid


# ### 4.3.3. The parameter grid for decision tree
# The hyperparameters we want to tune are:
# - min_samples_split
# - min_samples_leaf
# 
# See details of the meaning of the hyperparametes in [sklearn decision tree documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

# In[ ]:


param_grid = [{'clf__min_samples_split': [2, 10, 30],
               'clf__min_samples_leaf': [1, 10, 30]}]

param_grids['dt'] = param_grid


# ### 4.3.4. The parameter grid for random forest
# The hyperparameters we want to tune are:
# - n_estimators
# - min_samples_split
# - min_samples_leaf
# 
# See details of the meaning of the hyperparametes in [sklearn random forest documentation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

# In[ ]:


param_grid = [{'clf__n_estimators': [2, 10, 30],
               'clf__min_samples_split': [2, 10, 30],
               'clf__min_samples_leaf': [1, 10, 30]}]

param_grids['rf'] = param_grid


# ### 4.3.5. The parameter grid for support vector machine
# The hyperparameters we want to tune are:
# - C
# - gamma
# - kernel
# 
# See details of the meaning of the hyperparametes in [sklearn support vector machine documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

# In[ ]:


param_grid = [{'clf__C': [0.01, 0.1, 1, 10, 100],
               'clf__gamma': [0.01, 0.1, 1, 10, 100],
               'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]

param_grids['svc'] = param_grid


# ## 4.4. Hyperparameter tuning
# Here we use two functions for hyperparameter tuning:
# - [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html): Exhaustive search over specified parameter values for an estimator
# - [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html): Stratified K-Folds cross-validator

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# The list of [best_score_, best_params_, best_estimator_]
best_score_param_estimators = []

# For each classifier
for name in pipe_clfs.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipe_clfs[name],
                      param_grid=param_grids[name],
                      scoring='precision',
                      n_jobs=1,
                      cv=StratifiedKFold(n_splits=10,
                                         shuffle=True,
                                         random_state=0))
    # Fit the pipeline
    gs = gs.fit(X_train, y_train)
    
    # Update best_score_param_estimators
    best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])


# ## 4.5. Model selection

# In[ ]:


# Sort best_score_param_estimators in descending order of the best_score_
best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x : x[0], reverse=True)

# For each [best_score_, best_params_, best_estimator_]
for best_score_param_estimator in best_score_param_estimators:
    # Print out [best_score_, best_params_, best_estimator_], where best_estimator_ is a pipeline
    # Since we only print out the type of classifier of the pipeline
    print([best_score_param_estimator[0], best_score_param_estimator[1], type(best_score_param_estimator[2].named_steps['clf'])], end='\n\n')


# ## 5. Print the accuracy of the best model on the testing data

# In[ ]:


print(best_score_param_estimators[0][2].score(X_test, y_test))


# ### 7.1. Print the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix

y_pred = rf.predict(X_test)
confusion_matrix(y_test, y_pred, labels=[0, 1])


# In[ ]:


#Print Metrics Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# <h1 align="center"> 
# ROC CURVE
# </h1>

# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier as rf
logit_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

