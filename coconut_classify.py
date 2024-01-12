import matplotlib.pyplot as plt
import numpy as np
import os, time, datetime

from skelm import ELMClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from dataclass import DataClass

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

def plot_results(y_predict,y_test,classes,z_test,times=None,savename='',timestamp='',show_plot=False):
    os.makedirs(os.path.join("results"),exist_ok=True)
    assert len(y_test.shape) <= 2 and len(y_test.shape) == len(y_predict.shape)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    if len(y_predict.shape) == 2:
        y_predict = np.argmax(y_predict, axis=1)
    y_test,y_predict = y_test.astype(np.int32),y_predict.astype(np.int32)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    print(f"Accuracy: {metrics.accuracy_score(y_test,y_predict)}")
    print(f"Precision: {metrics.precision_score(y_test,y_predict,average=None)}")
    print(f"Recall: {metrics.recall_score(y_test,y_predict,average=None)}")
    print(f"F1 Score: {metrics.f1_score(y_test,y_predict,average=None)}")
    print(f"MCC: {metrics.matthews_corrcoef(y_test,y_predict)}")
    print(f"Balanced Accuracy: {metrics.balanced_accuracy_score(y_test,y_predict)}")
    disp.figure_.suptitle(f"Confusion Matrix for classifier\n{savename}")
    if show_plot:
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        disp.figure_.show()
        disp.figure_.waitforbuttonpress()
    disp.figure_.savefig(os.path.join("results",f"{savename}-confmat_{timestamp}.png"))
    plt.close() #clear the figure from memory
    with open(os.path.join("results",f"all-metrics_{timestamp}.txt"),'a') as f:
        f.write(f"Classifier: {savename}\n")
        f.write(metrics.classification_report(y_test, y_predict))
        f.write(f"\nTraining time (s): {times[0]}\nTesting time (s): {times[1]}\n")
    with open(os.path.join("results",f"metrics_{savename}_{timestamp}.txt"),'w') as f:
        f.write(f"Classifier: {savename}\n")
        f.write(metrics.classification_report(y_test, y_predict))
        f.write(f"\nTraining time (s): {times[0]}\nTesting time (s): {times[1]}\n")
    classes = {i:classes[i] for i in range(len(classes))}
    classmap = np.vectorize(classes.__getitem__)
    np.savetxt(os.path.join("results",f"{savename}_{timestamp}.csv"),np.stack((classmap(y_predict),classmap(y_test),z_test),axis=-1),delimiter=',',header="Predicted,Actual,Filename",fmt="%s")


def eval_classifier(classifier,X_train,X_test,y_train,y_test,classes,z_test,savename='',timestamp='',show_plot=False):
    print(f"Starting classifier {savename} evaluation...")
    t_start = time.time_ns()
    classifier.fit(X_train,y_train)
    t_train = time.time_ns()
    y_predict = classifier.predict(X_test)
    t_val = time.time_ns()
    times=[(t_train-t_start)*1e-9,(t_val-t_train)*1e-9]
    print(f"Evaluation of {savename} complete!")
    print(f"Training time: {times[0]} seconds | Testing time: {times[1]} seconds")
    plot_results(y_predict,y_test,classes,z_test,times,savename,timestamp=timestamp,show_plot=show_plot)
    print("\n\n")

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    train_aug_count,test_aug_count = 1,1
    limit=0 #set to 0 for the actual evaluation
    data_path = "datasets/"
    results_path = "results/"
    raw_path =  os.path.join(data_path,"Coconut Tree Disease Dataset/")
    split_path = os.path.join(data_path,"split/")
    processed_path = os.path.join(data_path,"processed/processed/")
    processed_train_path = os.path.join(data_path,"processed/processed/train/")
    processed_test_path = os.path.join(data_path,"processed/processed/test/")
    os.makedirs(raw_path,exist_ok=True)
    os.makedirs(results_path,exist_ok=True)
    #Read and process the dataset
    #random_state = np.random.randint((2**31)-1)
    random_state = 317401096 
    print(f"Random state used: {random_state}")
    with open(os.path.join(results_path,f"METRICS_{timestamp}.txt"),'w') as f:
        f.write(f"Seed used: {random_state}")

    if not os.path.exists(processed_path):
        data = DataClass(folder_name=raw_path,random_state=random_state)
        data.read_data(limit=limit)
        data.process_images(resize=True,segment_images=False)
        data.write_data(processed_path)
        data.write_split_data(split_path,test_size=0.3,train_size=0.7,random_state=random_state,shuffle=True if random_state is not None else False,stratify=data.labels)
        print("Split complete!")
    #data = DataClass(folder_name=processed_path)

    if not os.path.exists(processed_train_path) or not os.path.exists(processed_test_path):
        train_data = DataClass(folder_name=os.path.join(split_path,"train/"))
        test_data = DataClass(folder_name=os.path.join(split_path,"test/"))
        train_data.read_data()
        test_data.read_data()
        train_data.augment_images(train_aug_count)
        test_data.augment_images(test_aug_count)
        train_data.write_data(processed_train_path)
        train_data.write_data(processed_test_path)

    train_data = DataClass(folder_name=processed_train_path)
    test_data = DataClass(folder_name=processed_test_path)
    train_data.read_data()
    test_data.read_data()
    classes = train_data.classes

    print("Preprocessing data...")
    X_train,y_train = train_data.get_dataset()
    X_train1,y_train1 = train_data.apply_feature_extraction(X_train,y_train) #contains features
    X_train2,y_train2 = train_data.normalize_and_encode(X_train,y_train) #contains normalized images
    X_train1,y_train1 = train_data.balance_data(X_train1,y_train1,apply_oversampling=True) #balance dataset
    X_train2,y_train2 = train_data.balance_data(X_train2,y_train2,apply_oversampling=True) #balance dataset
    y_train1_encoded = train_data.encode_labels(y_train1) #encode labels
    y_train2_encoded = train_data.encode_labels(y_train2) #encode labels
    X_train1 = X_train1.astype(np.float64) #avoid overflow
    X_train2 = X_train2.astype(np.float64) #avoid overflow
    
    X_test,y_test = test_data.get_dataset()
    z_test = np.array(test_data.filenames)
    X_test1,y_test1 = test_data.apply_feature_extraction(X_test,y_test)
    X_test2,y_test2 = test_data.normalize_and_encode(X_test,y_test)
    y_test1_encoded = test_data.encode_labels(y_test1) #encode labels
    y_test2_encoded = test_data.encode_labels(y_test2) #encode labels
    X_test1 = X_test1.astype(np.float64) #avoid overflow
    X_test2 = X_test2.astype(np.float64) #avoid overflow
    print("Preprocessing complete!")

    print("Loading testbench classifiers...")
    elm_classifiers = [[ELMClassifier(n_neurons=num_hidden,ufunc=ufunc,random_state=random_state),f"elm-{num_hidden}_{ufunc.__name__}"]\
                        for num_hidden in [100,500,1000,5000]\
                        for ufunc in [tanh_f,lrelu_f,linear_f]]
                        #for num_hidden in [100,500,1000,5000,10000]\
                        #for ufunc in [tanh_f,log_f,swish_f,lrelu_f,elu_f,linear_f]]
    knn_classifiers = [[KNeighborsClassifier(i),f"knn_{i}"] for i in range(3,11)]
    svm_classifiers = [[SVC(kernel=kernel, C=0.025, random_state=random_state,decision_function_shape=shape),f"svm_{kernel}_shape"]\
                        for kernel in ["rbf","linear","poly"]\
                        for shape in ["ovo","ovr"]]
    disc_classifiers = [[LinearDiscriminantAnalysis(),f"linear_discriminant_analysis"],[QuadraticDiscriminantAnalysis(),f"quadratic_discriminant_analysis"]]
    gauss_classifiers= [[GaussianNB(),f"gaussian_naivebayes"]]

    classifiers = elm_classifiers+knn_classifiers+svm_classifiers+disc_classifiers+gauss_classifiers

    for idx in range(len(classifiers)):
        print(f"Evaluating classifier {(idx*2)+1}/{len(classifiers)*2}: {classifiers[idx][1]}")
        eval_classifier(classifiers[idx][0],X_train1,X_test1,y_train1,y_test1,classes,z_test,classifiers[idx][1]+"-feature",timestamp,False)
        print(f"Evaluating classifier {(idx*2)+2}/{len(classifiers)*2}: {classifiers[idx][1]}")
        eval_classifier(classifiers[idx][0],X_train2,X_test2,y_train2,y_test2,classes,z_test,classifiers[idx][1]+"-normalized",timestamp,False)


