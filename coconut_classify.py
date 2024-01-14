import matplotlib.pyplot as plt
import numpy as np
import os, time, datetime, gc

from skelm import ELMClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from dataclass import DataClass

#ACTIVATION FUNCTIONS USED

#define the logistic sigmoid function with a=2.0
def log_f(v,a=2.0) : #phi(v) = 1/(1+e^(-a*v)) = (e^(a*v))/((e^(a*v))+1)
    return np.where(v>=0, 1/(1+np.exp(-a*v)), np.exp(a*v)/(np.exp(a*v)+1))

#define the swish function with b=1.0
def swish_f(v,b=1.0) : #phi(v) = v/(1+e^(-b*v)) = v(e^(b*v))/((e^(b*v))+1)
    return v*np.where(v>=0, 1/(1+np.exp(-b*v)), np.exp(b*v)/(np.exp(b*v)+1))

#define the leaky ReLU function with gradient parameter a=0.01
def lrelu_f(v,a=0.01): #phi(v) = v if v > 0, a*v if v < 0
    return np.where(v>0, v,a*v)

activation_functions = {"Logistic":log_f,"Swish":swish_f,"Leaky ReLU":lrelu_f}


def plot_results(y_predict,y_test,classes,filenames,times=None,savename='',timestamp='',show_plot=False):
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
    np.savetxt(os.path.join("results",f"{savename}_{timestamp}.csv"),np.stack((classmap(y_predict),classmap(y_test),filenames),axis=-1),delimiter=',',header="Predicted,Actual,Filename",fmt="%s")


def eval_classifier(classifier,X_train,X_test,y_train,y_test,classes,filenames,savename='',timestamp='',show_plot=False):
    print(f"Starting classifier {savename} evaluation...")
    t_start = time.time_ns()
    classifier.fit(X_train,y_train)
    t_train = time.time_ns()
    y_predict = classifier.predict(X_test)
    t_val = time.time_ns()
    times=[(t_train-t_start)*1e-9,(t_val-t_train)*1e-9]
    print(f"Evaluation of {savename} complete!")
    print(f"Training time: {times[0]} seconds | Testing time: {times[1]} seconds")
    plot_results(y_predict,y_test,classes,filenames,times,savename,timestamp=timestamp,show_plot=show_plot)
    print("\n\n")

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    limit=0 #set to 0 for the actual evaluation
    data_path = "datasets/"
# Dataset is from https://data.mendeley.com/datasets/gh56wbsnj5/1
    data_name = "Coconut Tree Disease Dataset"
# Dataset is from https://www.kaggle.com/datasets/samitha96/coconutdiseases
#    data_name = "Coconut Leaf Dataset for Pest Identification"
    results_path = os.path.join("results/",data_name)
    raw_path = os.path.join(data_path,data_name)
    split_path = os.path.join(data_path,"processed/",data_name)
    processed_path = os.path.join(data_path,"processed/",data_name)
    processed_train_path = os.path.join(processed_path,"train/")
    processed_test_path = os.path.join(processed_path,"test/")
    cached_data_path = os.path.join(processed_path,"cached_data/",data_name) #path for feature dataset
    os.makedirs(raw_path,exist_ok=True)
    os.makedirs(results_path,exist_ok=True)
    #Read and process the dataset
    #random_state = np.random.randint((2**31)-1)
    random_state = 317401096
    print(f"Random state used: {random_state}")
    with open(os.path.join(results_path,f"METRICS_{data_name}{timestamp}.txt"),'w') as f:
        f.write(f"Seed used: {random_state}")


    use_cache = os.path.exists(cached_data_path)
    for colorspace in ['rgb','hsv','cielab','gray']:
        for use_lbp in [True,False]:
            feature_name = f"{colorspace}{'_lbp' if use_lbp else ''}"
            #Evaluate all classifiers on feature dataset
            if not os.path.isfile(os.path.join(cached_data_path,f"{feature_name}.npz")):
                use_cache = False

    if not use_cache: #regenerate cache if not complete
        print("Reading data...")
        if not os.path.exists(processed_path):
            data = DataClass(folder_name=raw_path)
            data.read_data(limit=limit)
            data.process_images(resize=True,segment_images=False)
            data.write_split_data(split_path,test_size=0.3,train_size=0.7,random_state=random_state,stratify=data.labels)
            print("Split complete!")
            del data #clear object once no longer used
            gc.collect()
        print("Reading training data...")
        train_data = DataClass(folder_name=processed_train_path)
        print("Reading testing data...")
        test_data = DataClass(folder_name=processed_test_path)
        train_data.read_data()
        test_data.read_data()
        classes = train_data.classes
        filenames = np.array(test_data.filenames)

        print("Preprocessing data...")
        X_train,y_train = train_data.get_dataset()
        X_test,y_test = test_data.get_dataset()
        X_train,y_train = train_data.balance_data(X_train,y_train,apply_undersampling=True,random_state = random_state) #balance training dataset before using
        print("Balanced training set!")
        os.makedirs(cached_data_path,exist_ok=True)
        for colorspace in ['rgb','hsv','cielab','gray']:
            for use_lbp in [True,False]:
                feature_name = f"{colorspace}{'_lbp' if use_lbp else ''}"
                X_train1,y_train1 = train_data.apply_feature_extraction(X_train,y_train,colorspace=colorspace,use_lbp=use_lbp) #contains features
                X_test1,y_test1 = test_data.apply_feature_extraction(X_test,y_test,colorspace=colorspace,use_lbp=use_lbp)
                print(X_train1.shape)
                print(y_train1.shape)
                print(X_test1.shape)
                print(y_test1.shape)
            print(f"Preprocessing dataset {feature_name} complete!")
            np.savez(os.path.join(cached_data_path,f"{feature_name}.npz"),X_train1=X_train1,y_train1=y_train1,X_test1=X_test1,y_test1=y_test1,classes=classes,filenames=filenames)
            print("Saved preprocessed data as cache")
        #Clear unused data classes
        del train_data, test_data, X_train, X_test, y_train, y_test
        #Clear other data classes; reload them later
        del X_train1,X_test1,y_train1,y_test1,classes,filenames,feature_name
        gc.collect()
    print("Pre-processing stage complete!")

    for ufunc,func_name in activation_functions.items():
        for num_hidden in range(500,10500,500):
            classifier,classifier_name = ELMClassifier(n_neurons=num_hidden,ufunc=ufunc,random_state=random_state),f"elm-{num_hidden}_{func_name}"
        for colorspace in ['rgb','hsv','cielab','gray']:
                for use_lbp in [True,False]:
                    feature_name = f"{colorspace}{'_lbp' if use_lbp else ''}"
                    #Evaluate all classifiers on feature dataset
                    loaded_data = np.load(os.path.join(cached_data_path,f"{feature_name}.npz"),allow_pickle=True)
                    X_train1,X_test1,y_train1,y_test1,classes,filenames = loaded_data['X_train1'],loaded_data['X_test1'],loaded_data['y_train1'],loaded_data['y_test1'],loaded_data['classes'],loaded_data['filenames']
                    del loaded_data
                    gc.collect()
                    print(f"Evaluating classifier {classifier_name} on {feature_name}")
                    print(f"{X_train.shape[1]}={X_test.shape[1]} features | {X_train.shape[0]}={y_train.shape} training points | {X_test.shape[0]}={y_test.shape} testing points")
                    eval_classifier(classifier,X_train1,X_test1,y_train1,y_test1,classes,filenames,f"{data_name}{classifier_name}_{feature_name}",timestamp,False)

#To do: automate loading of alternate dataset
#To do: automate plotting of accuracy metrics against other classifiers