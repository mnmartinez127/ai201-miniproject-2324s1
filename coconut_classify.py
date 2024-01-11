import matplotlib.pyplot as plt
import numpy as np
import os, time

from skelm import ELMClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from data.dataclass import DataClass

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

#Linear function
def linear_f(v): #phi(v) = v
    return v

def plot_results(y_predict,y_test,times=None,savename="",show_plot=False):
    os.makedirs(os.path.join("Result"),exist_ok=True)
    y_test = np.argmax(y_test, axis=1)
    y_predict = np.argmax(y_predict, axis=1)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    disp.figure_.suptitle(f"Confusion Matrix for classifier\n{savename}")
    if show_plot:
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        disp.figure_.show()
        disp.figure_.waitforbuttonpress()
    disp.figure_.savefig(os.path.join("Result",f"{savename}-confmat.png"))
    disp.figure_.clear()
    with open(os.path.join("Result",f"{savename}-metrics.txt"),'w') as f:
        f.write(f"\nTraining time (s): {times[0]}")
        f.write(f"\nTesting time (s): {times[1]}")
        f.write(metrics.classification_report(y_test, y_predict))
    np.savetxt(os.path.join("Result",f"{savename}-predict.txt"),y_predict)
    np.savetxt(os.path.join("Result",f"{savename}-test.txt"),y_test)


def eval_classifier(classifier,X_train,X_test,y_train,y_test,savename="",show_plot=False):
    print("Starting classifier evaluation...")
    t_start = time.time_ns()
    classifier.fit(X_train,y_train)
    t_train = time.time_ns()
    y_predict = classifier.predict(X_test)
    t_val = time.time_ns()
    print(f"Evaluation of {savename} complete!")
    plot_results(y_predict,y_test,[(t_train-t_start)*1e-9,(t_val-t_train)*1e-9],savename,show_plot=show_plot)

if __name__ == "__main__":

    random_state = 13535
    train_aug_count,test_aug_count = 5,3
    limit=0
    raw_path = 'data/datasets/Coconut Tree Disease Dataset/'
    processed_path = 'data/datasets/processed/processed/'
    split_path = 'data/datasets/split/'
    processed_train_path = 'data/datasets/processed/train/'
    processed_test_path = 'data/datasets/processed/test/'
    #Read and process the dataset

    if not os.path.exists(processed_path) or not os.path.exists(processed_train_path) or not os.path.exists(processed_test_path):
        data = DataClass(folder_name=raw_path,random_state=random_state)
        data.read_data(limit=limit)
        data.process_images(resize=True)
        data.write_data(processed_path)
        data.write_split_data(split_path,test_size=0.3,train_size=0.7,random_state=random_state,shuffle=True if random_state is not None else False,stratify=True)
    data = DataClass(folder_name=processed_path)
    train_data = DataClass(folder_name=os.path.join(split_path,"/train/"))
    test_data = DataClass(folder_name=os.path.join(split_path,"/test/"))

    train_data.augment_images(train_aug_count)
    test_data.augment_images(test_aug_count)
    train_data.write_data(processed_train_path)
    train_data.write_data(processed_test_path)
    train_data = DataClass(folder_name=processed_train_path)
    test_data = DataClass(folder_name=processed_test_path)

    X_train,y_train = train_data.get_dataset()
    X_train1,y_train1 = train_data.apply_feature_extraction(X_train,y_train) #contains features
    X_train2,y_train2 = train_data.normalize_and_encode(X_train,y_train) #contains normalized images
    X_train1,y_train1 = train_data.balance_data(X_train1,y_train1,apply_smote=True) #balance dataset
    X_train2,y_train2 = train_data.balance_data(X_train2,y_train2,apply_smote=True) #balance dataset
    print(X_train.shape)
    print(y_train.shape)

    X_test,y_test = test_data.get_dataset()
    X_test1,y_test1 = test_data.apply_feature_extraction(X_test,y_test)
    X_test2,y_test2 = test_data.normalize_and_encode(X_test,y_test)
    z_test = np.array(test_data.filenames)
    print(X_test.shape)
    print(y_test.shape)
    print(z_test.shape)

    elm_classifiers = [[ELMClassifier(n_neurons=num_hidden,ufunc=ufunc,random_state=random_state),f"elm-{num_hidden}_{ufunc.__name__}"]\
                        for num_hidden in [100,1000]\
                        for ufunc in [tanh_f,log_f,swish_f,lrelu_f,elu_f,linear_f]]
    knn_classifiers = [[KNeighborsClassifier(i),f"knn_{i}"] for i in range(3,11)]
    svm_classifiers = [[SVC(kernel=kernel, C=0.025, random_state=random_state),f"svm_{kernel}"] for kernel in ["rbf","linear","poly"]],
    adaboost_classifiers = [[AdaBoostClassifier(n_estimators=n,random_state=random_state),f"adaboost_{n}"] for n in range(10,101,10)]
    disc_classifiers = [[LinearDiscriminantAnalysis(),f"lda"],[QuadraticDiscriminantAnalysis(),f"qda"]]
    gauss_classifiers= [[GaussianNB()],f"gaussian_naivebayes"]

    eval_classifier(elm_classifiers[0][0],X_train1,X_test1,y_train1,y_test1,elm_classifiers[0][1]+"-feature",True)
    eval_classifier(elm_classifiers[0][0],X_train1,X_test1,y_train1,y_test1,elm_classifiers[0][1]+"-normalized",True)
    eval_classifier(svm_classifiers[0][0],X_train1,X_test1,y_train1,y_test1,svm_classifiers[0][1]+"-feature",True)
    eval_classifier(svm_classifiers[0][0],X_train1,X_test1,y_train1,y_test1,svm_classifiers[0][1]+"-normalized",True)

