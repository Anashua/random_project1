import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.write("""
# EXPLORING CLASSIFIERS
""")
dataset_name =st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Quality"))
st.write(dataset_name)
classifier_name=st.sidebar.selectbox("Select Classifier",("Linear Regression","SVM","RANDOM FOREST"))
def get_dataset(dataset_name):
    if dataset_name=="Iris":
        my_data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        my_data=datasets.load_breast_cancer()
    else:
        my_data=datasets.load_wine()
    X=my_data.data
    y=my_data.target
    return X,y
X,y =get_dataset(dataset_name)
st.write("shape of the dataset",X.shape)
st.write("number of classes",len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"]=C
    elif clf_name=="RANDOM FOREST":
        max_depth=st.sidebar.slider("max_depth",2,15)
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators
    else:
        params=None
    return params
params=add_parameter_ui(classifier_name)
def get_classifier(clf_name,params):
    if clf_name=="SVM":
        clf=SVC(C=params["C"])  
    elif clf_name =="Linear Regression" :
        clf=LinearRegression()
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf
clf=get_classifier(classifier_name,params)

# Classification
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_predict=clf.predict(X_test)

def get_pred(classifier_name):
    if classifier_name=="Linear Regression":
        st.write('R-score is: ',clf.score(X_test,y_test))

    else:
        accuracy=accuracy_score(y_test,y_predict)
        st.write("classifier =",classifier_name)
        st.write("accuracy=",accuracy)
get_pred(classifier_name)

#PLOT
pca=PCA(2)
X_projected=pca.fit_transform(X)

x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar()

st.pyplot(fig)

