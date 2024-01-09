
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from skelm import ELMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from PIL import Image

#ACTIVATION FUNCTIONS
#define the hyperbolic tangent function with a=1.716 and b=2/3
def tanh_f(v,a=1.716,b=(2.0/3.0)): #phi(v) = a*tanh(b*v)
    return a*np.tanh(b*v)

#define the logistic sigmoid function with a=2.0
def log_f(v,a=2.0) : #phi(v) = 1/(1+e^(-a*v)) = (e^(a*v))/((e^(a*v))+1)
    return np.where(v>=0, 1/(1+np.exp(-a*v)), np.exp(a*v)/(np.exp(a*v)+1))

#define the swish function with b=1.0
def swish_f(v,b=1.0) : #phi(v) = v/(1+e^(-b*v)) = v(e^(b*v))/((e^(b*v))+1)
    return v*np.where(v>=0, 1/(1+np.exp(-b*v)), np.exp(b*v)/(np.exp(b*v)+1))

#define the leaky ReLU function with gradient parameter a=0.01
def lrelu_f(v,a=0.01): #phi(v) = v if v > 0, a*v if v < 0
    return np.where(v>0, v,a*v)

#define the ELU function with gradient parameter a=0.01
def elu_f(v,a=1.67326,l=1.0507): #phi(v) = lv if v >= 0, la(e^v - 1) if v < 0
    return np.where(v>=0, l*v,l*a*(np.exp(v)-1))

def linear_f(v): #phi(v) = v
    return v

def plot_results(y_predict,y_test,times,savename=""):
    os.makedirs(os.path.join("Result"),exist_ok=True)
    y_test = np.argmax(y_test, axis=1)
    y_predict = np.argmax(y_predict, axis=1)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    disp.figure_.suptitle(f"Confusion Matrix for classifier\n{savename}")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    #disp.figure_.show()
    #disp.figure_.waitforbuttonpress()
    disp.figure_.savefig(os.path.join("Result",f"{savename}-confmat.png"))
    disp.figure_.clear()
    with open(os.path.join("Result",f"{savename}-metrics.txt"),'w') as f:
        f.write(metrics.classification_report(y_test, y_predict))
        f.write(f"\nTraining time: {times[0]} s")
        f.write(f"\nTesting time: {times[1]} s")
    np.savetxt(os.path.join("Result",f"{savename}-predict.txt"),y_predict)
    np.savetxt(os.path.join("Result",f"{savename}-test.txt"),y_test)


def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    labels = []
    for class_id, label in enumerate(os.listdir(folder)):
        class_folder = os.path.join(folder, label)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                with Image.open(image_path) as img:
                    img = img.resize(target_size)
                    images.append(np.array(img))
                    labels.append(class_id)
    return np.array(images), np.array(labels)


def preprocess_data(X, y):
    X = X.reshape(X.shape[0], -1)
    X = X.astype('float32') / 255  # normalize to range [0, 1]
    lb = LabelBinarizer()
    y = lb.fit_transform(y)  # one-hot encode the labels
    return X, y


def eval_classifier(classifier,savename=""):
    X,y = load_images_from_folder('preprocessing/datasets/processed')
    X,y = preprocess_data(X, y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.7,train_size=0.3,random_state=random_state,shuffle=True if random_state is not None else False)
    t_start = time.time_ns()
    classifier.fit(X_train,y_train)
    t_train = time.time_ns()
    y_predict = classifier.predict(X_test)
    t_val = time.time_ns()
    plot_results(y_predict,y_test,[(t_train-t_start)*1e-9,(t_val-t_train)*1e-9],savename)
    print(f"Evaluation of {savename} complete!")


load_data = load_digits
random_state = 13535
elm_classifiers = [[ELMClassifier(n_neurons=num_hidden,ufunc=ufunc,random_state=random_state),f"elm-{num_hidden}_{ufunc.__name__}"]\
                    for num_hidden in [100,1000]\
                    for ufunc in [tanh_f,log_f,swish_f,lrelu_f,elu_f,linear_f]]
knn_classifiers = [[KNeighborsClassifier(i),f"knn_{i}"] for i in range(3,11)]
svm_classifiers = [[SVC(kernel=kernel, C=0.025, random_state=random_state),f"svm_{kernel}"] for kernel in ["rbf","linear","poly"]],
adaboost_classifiers = [[AdaBoostClassifier(n_estimators=n,random_state=random_state),f"adaboost_{n}"] for n in range(10,101,10)]
disc_classifiers = [[LinearDiscriminantAnalysis(),f"lda"],[QuadraticDiscriminantAnalysis(),f"qda"]]

for clf,savename in elm_classifiers:
    eval_classifier(clf, savename)