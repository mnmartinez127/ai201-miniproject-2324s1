import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, time, datetime

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


def plot_results(y_predict,y_test,classes,filenames,times=None,savepath='',savename='',timestamp='',show_plot=False):
    os.makedirs(savepath,exist_ok=True)
    os.makedirs(os.path.join(savepath,"confmat"),exist_ok=True)
    os.makedirs(os.path.join(savepath,"predictions"),exist_ok=True)
    os.makedirs(os.path.join(savepath,"metrics"),exist_ok=True)
    os.makedirs(savepath,exist_ok=True)
    os.makedirs(savepath,exist_ok=True)
    assert len(y_test.shape) <= 2 and len(y_test.shape) == len(y_predict.shape)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    if len(y_predict.shape) == 2:
        y_predict = np.argmax(y_predict, axis=1)
    y_test,y_predict = y_test.astype(np.int32),y_predict.astype(np.int32)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    accuracy = metrics.accuracy_score(y_test,y_predict)
    precision = metrics.precision_score(y_test,y_predict,average=None)
    recall = metrics.recall_score(y_test,y_predict,average=None)
    f1_score = metrics.f1_score(y_test,y_predict,average=None)
    mcc = metrics.matthews_corrcoef(y_test,y_predict)
    bal_accuracy = metrics.balanced_accuracy_score(y_test,y_predict)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"MCC: {mcc}")
    print(f"Balanced Accuracy: {bal_accuracy}")
    disp.figure_.suptitle(f"Confusion Matrix for classifier\n{savename}")
    if show_plot:
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        disp.figure_.show()
        disp.figure_.waitforbuttonpress()
    disp.figure_.savefig(os.path.join(savepath,"confmat",f"{savename}-confmat_{timestamp}.png"))
    plt.close() #clear the figure from memory

    with open(os.path.join(savepath,f"all-metrics_{timestamp}.txt"),'a') as f: #metric summary for comparing results
        f.write(f"Classifier: {savename}\n")
        f.write(f"\nAccuracy: {accuracy}")
        f.write(f"\nBalanced Accuracy: {bal_accuracy}")
        f.write(f"\nF1 Score: {f1_score}")
        f.write(f"\nMCC: {mcc}")
        f.write(f"\n\nTraining time (s): {times[0]}\nTesting time (s): {times[1]}\n\n")
    with open(os.path.join(savepath,"metrics",f"metrics_{savename}_{timestamp}.txt"),'w') as f:
        f.write(f"Classifier: {savename}\n")
        f.write(metrics.classification_report(y_test, y_predict))
        f.write(f"\nF1 Score: {f1_score}")
        f.write(f"\nMCC: {mcc}")
        f.write(f"\nClasses: {classes}")
        f.write(f"\nPrecision: {precision}")
        f.write(f"\nRecall: {recall}")
        f.write(f"\n\nTraining time (s): {times[0]}\nTesting time (s): {times[1]}\n\n")

    classes = {i:classes[i] for i in range(len(classes))}
    classmap = np.vectorize(classes.__getitem__)
    np.savetxt(os.path.join(savepath,"predictions",f"{savename}_{timestamp}.csv"),np.stack((classmap(y_predict),classmap(y_test),filenames),axis=-1),delimiter=',',header="Predicted,Actual,Filename",fmt="%s")
    print("\n\n")
    return accuracy,bal_accuracy,times[0],times[1]

def eval_classifier(classifier,X_train,X_test,y_train,y_test,classes,filenames,savepath='',savename='',timestamp='',show_plot=False):
    print(f"Starting classifier {savename} evaluation...")
    t_start = time.time_ns()
    classifier.fit(X_train,y_train)
    t_train = time.time_ns()
    y_predict = classifier.predict(X_test)
    t_val = time.time_ns()
    times=[(t_train-t_start)*1e-9,(t_val-t_train)*1e-9]
    print(f"Evaluation of {savename} complete!")
    print(f"Training time: {times[0]} seconds | Testing time: {times[1]} seconds")
    return plot_results(y_predict,y_test,classes,filenames,times,savepath,savename,timestamp=timestamp,show_plot=show_plot)



if __name__ == "__main__":
    # Dataset is from https://data.mendeley.com/datasets/gh56wbsnj5/1
    # data_name = "Coconut Tree Disease Dataset"
    # Dataset is from https://www.kaggle.com/datasets/samitha96/coconutdiseases
    # data_name = "Coconut Leaf Dataset for Pest Identification"
    # Combination dataset combines train and test sets from both datasets. It must be manually combined.
    datasets_name = ["Coconut Tree Disease Dataset","Coconut Leaf Dataset for Pest Identification","Combination Dataset"]
    for data_name in datasets_name:
        timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        limit=0 #set to 0 for the actual evaluation
        data_path = "datasets/"
        results_path = os.path.join("results/",data_name)
        raw_path = os.path.join(data_path,data_name)
        split_path = os.path.join(data_path,"processed/",data_name)
        processed_path = os.path.join(data_path,"processed/",data_name)
        processed_train_path = os.path.join(processed_path,"train/")
        processed_test_path = os.path.join(processed_path,"test/")
        cached_data_path = os.path.join(processed_path,"cached_data/") #path for feature dataset
        os.makedirs(results_path,exist_ok=True)
        #Read and process the dataset
        #random_state = np.random.randint((2**31)-1)
        random_state = 317401096

        use_cache = os.path.exists(cached_data_path)
        for colorspace in ['rgb','hsv','cielab']:
            for use_lbp in [False,True]:
                feature_name = f"{colorspace}{'_lbp' if use_lbp else ''}"
                #Evaluate all classifiers on feature dataset
                if not os.path.isfile(os.path.join(cached_data_path,f"{feature_name}.npz")):
                    use_cache = False

        if not use_cache: #regenerate cache if not complete
            print(f"Seed used: {random_state}")
            with open(os.path.join(results_path,f"seed-{random_state}_{data_name}{timestamp}.txt"),'w') as f:
                f.write(f"Seed used: {random_state}")

            print("Reading data...")
            if not os.path.exists(processed_path) and os.path.exists(raw_path):
                data = DataClass(folder_name=raw_path)
                data.read_data(limit=limit)
                data.process_images(resize=True,segment_images=False)
                data.write_split_data(split_path,test_size=0.2,train_size=0.8,random_state=random_state,stratify=data.labels)
                print("Split complete!")
                del data #clear object once no longer used
            if os.path.exists(processed_path):
                print("Reading training data...")
                train_data = DataClass(folder_name=processed_train_path)
                train_data.read_data()
                print("Reading testing data...")
                test_data = DataClass(folder_name=processed_test_path)
                test_data.read_data()
                classes = train_data.classes
                filenames = np.array(test_data.filenames)

                print("Preprocessing data...")
                X_train,y_train = train_data.get_dataset()
                X_test,y_test = test_data.get_dataset()

                train_counts = np.unique(y_train,return_counts=True)
                test_counts = np.unique(y_test,return_counts=True)
                print(f"Training Classes:{classes}\n")
                print(f"Training Samples: {X_train.shape[0]}={y_train.shape[0]}\n")
                print('\n'.join([f"{train_counts[0][i]}: {train_counts[1][i]}" for i in range(len(train_counts[0]))]))
                print(f"\n\n")
                print(f"Testing Samples: {X_train.shape[0]}={y_train.shape[0]}\n")
                print('\n'.join([f"{classes[train_counts[0][i]]}: {train_counts[1][i]}" for i in range(len(test_counts[0]))]))
                print(f"\n\n")
                with open(os.path.join(results_path,f"dataset_{data_name}{timestamp}.txt"),'w') as f:
                    f.write(f"Training Classes:{classes}\n")
                    f.write(f"Training Samples: {X_train.shape[0]}={y_train.shape[0]}\n")
                    f.write('\n'.join([f"{classes[train_counts[0][i]]}: {train_counts[1][i]}" for i in range(len(train_counts[0]))]))
                    f.write(f"\n\n")
                    f.write(f"Testing Samples: {X_train.shape[0]}={y_train.shape[0]}\n")
                    f.write('\n'.join([f"{classes[test_counts[0][i]]}: {test_counts[1][i]}" for i in range(len(test_counts[0]))]))
                    f.write(f"\n\n")


                X_train,y_train = train_data.balance_data(X_train,y_train,method='undersample',random_state = random_state) #balance training dataset before using

                bal_train_counts = np.unique(y_train,return_counts=True)
                print(f"Balanced Training Samples: {X_train.shape[0]}={y_train.shape[0]}\n")
                print('\n'.join([f"{classes[bal_train_counts[0][i]]}: {bal_train_counts[1][i]}" for i in range(len(bal_train_counts[0]))]))
                print(f"\n\n")
                with open(os.path.join(results_path,f"dataset_{data_name}{timestamp}.txt"),'a') as f:
                    f.write(f"Balanced Training Samples: {X_train.shape[0]}={y_train.shape[0]}\n")
                    f.write('\n'.join([f"{classes[bal_train_counts[0][i]]}: {bal_train_counts[1][i]}" for i in range(len(bal_train_counts[0]))]))
                    f.write(f"\n\n")

                print("Balanced training set!")
                os.makedirs(cached_data_path,exist_ok=True)
                for colorspace in ['rgb','hsv','cielab']:
                    for use_lbp in [False,True]:
                        feature_name = f"{colorspace}{'_lbp' if use_lbp else ''}"
                        X_train1,y_train1 = train_data.apply_feature_extraction(X_train,y_train,colorspace=colorspace,use_lbp=use_lbp) #contains features
                        X_test1,y_test1 = test_data.apply_feature_extraction(X_test,y_test,colorspace=colorspace,use_lbp=use_lbp)
                        print(f"Preprocessing dataset {feature_name} complete!")
                        np.savez(os.path.join(cached_data_path,f"{feature_name}.npz"),X_train1=X_train1,y_train1=y_train1,X_test1=X_test1,y_test1=y_test1,classes=classes,filenames=filenames)

                        print(f"{feature_name} feature size: {X_train1.shape}\n")
                        with open(os.path.join(results_path,f"dataset_{data_name}{timestamp}.txt"),'a') as f:
                            f.write(f"{feature_name} feature size: {X_train1.shape[1]}={X_test1.shape[1]}\n")


                print("Saved preprocessed data as cache")
                #Clear unused data classes
                del train_data, test_data, X_train, X_test, y_train, y_test
                #Clear other data classes; reload them later
                del X_train1,X_test1,y_train1,y_test1,classes,filenames,feature_name
        print("Pre-processing stage complete!")

        if not os.path.isfile(os.path.join(results_path,"result-accuracy.csv")):
            perf = {'lbp':[],'activation function':[],'color space':[],'hidden nodes':[],'accuracy':[],'balanced accuracy':[],'training time':[],'testing time':[]}
            for func_name,ufunc in activation_functions.items():
                for num_hidden in range(1000,10001,500):
                    classifier,classifier_name = ELMClassifier(n_neurons=num_hidden,ufunc=ufunc,random_state=random_state),f"elm-{num_hidden}_{func_name}"
                    for colorspace in ['rgb','hsv','cielab']:
                        for use_lbp in [False,True]:
                            feature_name = f"{colorspace}{'_lbp' if use_lbp else ''}"
                            #Evaluate all classifiers on feature dataset
                            loaded_data = np.load(os.path.join(cached_data_path,f"{feature_name}.npz"),allow_pickle=True)
                            X_train1,X_test1,y_train1,y_test1,classes,filenames = loaded_data['X_train1'],loaded_data['X_test1'],loaded_data['y_train1'],loaded_data['y_test1'],loaded_data['classes'],loaded_data['filenames']
                            del loaded_data
                            print(f"Evaluating classifier {classifier_name} on {feature_name}")
                            print(f"{X_train1.shape[1]}={X_test1.shape[1]} features | {X_train1.shape[0]}={y_train1.shape[0]} training points | {X_test1.shape[0]}={y_test1.shape[0]} testing points")
                            accuracy,bal_accuracy,training_time,testing_time = eval_classifier(classifier,X_train1,X_test1,y_train1,y_test1,classes,filenames,savepath=results_path,savename=f"{classifier_name}_{feature_name}",timestamp=timestamp,show_plot=False)

                            perf['lbp'].append(use_lbp)
                            perf['activation function'].append(func_name)
                            perf['color space'].append(colorspace)
                            perf['hidden nodes'].append(num_hidden)
                            perf['accuracy'].append(accuracy)
                            perf['balanced accuracy'].append(bal_accuracy)
                            perf['training time'].append(training_time)
                            perf['testing time'].append(testing_time)
            perf = pd.DataFrame(perf)
            perf = perf.sort_values(by=["lbp","activation function","color space","hidden nodes"],ignore_index=True)
            perf.to_csv(os.path.join(results_path,"result-accuracy.csv"))
        print("Evaluation stage complete!")

        print("Generating comparison graphs")
        perf = pd.read_csv(os.path.join(results_path,"result-accuracy.csv"),index_col=[0])
        perf = perf.sort_values(by=["lbp","activation function","color space","hidden nodes"],ignore_index=True)
        perf = perf[perf["hidden nodes"]%1000 == 0]
        perf.to_csv(os.path.join(results_path,"result-accuracy_1000.csv"))
        act_lines = {False:{"Leaky ReLU":'r',"Logistic":'g',"Swish":'b'},True:{"Leaky ReLU":'c',"Logistic":'y',"Swish":'m'}}
        color_lines = {"hsv":"-","rgb":":","cielab":"--","gray":"-."}

        fig1,ax1=plt.subplots()
        ax1.set_title(f"Accuracy: Color Histogram Features")
        ax1.set_xlim([0,int(perf["hidden nodes"].max())+1]) #establish range of values
        ax1.set_ylim([0,1])
        ax1.set_xlabel("Number of hidden nodes")
        ax1.set_ylabel("Accuracy (s)")

        fig2,ax2=plt.subplots()
        ax2.set_title(f"Accuracy: Color Histogram + LBP Features")
        ax2.set_xlim([0,int(perf["hidden nodes"].max())+1]) #establish range of values
        ax2.set_ylim([0,1])
        ax2.set_xlabel("Number of hidden nodes")
        ax2.set_ylabel("Accuracy (s)")

        fig3,ax3=plt.subplots()
        ax3.set_title(f"Accuracy: Best Parameters")
        ax3.set_xlim([0,int(perf["hidden nodes"].max())+1]) #establish range of values
        #manually set limit conditions
        ax3.set_ylim([min((perf[(perf["activation function"]!="Logistic") & (perf["color space"]!="cielab")]["balanced accuracy"])),1])
        ax3.set_xlabel("Number of hidden nodes")
        ax3.set_ylabel("Accuracy (s)")

        fig4,ax4=plt.subplots()
        ax4.set_title(f"Training time: Color Histogram Features")
        ax4.set_xlim([0,int(perf["hidden nodes"].max())+1]) #establish range of values
        ax4.set_ylim([0,(int(perf["training time"].max()+1)*1.1)])
        ax4.set_xlabel("Number of hidden nodes")
        ax4.set_ylabel("Training time (s)")

        fig5,ax5=plt.subplots()
        ax5.set_title(f"Testing time: Color Histogram Features")
        ax5.set_xlim([0,int(perf["hidden nodes"].max())+1]) #establish range of values
        ax5.set_ylim([0,(int(perf["testing time"].max()+1)*1.1)])
        ax5.set_xlabel("Number of hidden nodes")
        ax5.set_ylabel("Testing time (s)")


        perf_groups = perf.groupby(["activation function","lbp","color space"])
        for name,group in perf_groups:
            print(name)
            print(group)
            if name[1] == False:
                ax1.plot(group["hidden nodes"],group["balanced accuracy"],f'{act_lines[name[1]][name[0]]}o{color_lines[name[2]]}',label=f"{name[0]}-{name[2]}")
                ax4.plot(group["hidden nodes"],group["training time"],f'{act_lines[name[1]][name[0]]}o{color_lines[name[2]]}',label=f"{name[0]}-{name[2]}")
                ax5.plot(group["hidden nodes"],group["testing time"],f'{act_lines[name[1]][name[0]]}o{color_lines[name[2]]}',label=f"{name[0]}-{name[2]}")
            else:
                ax2.plot(group["hidden nodes"],group["balanced accuracy"],f'{act_lines[name[1]][name[0]]}o{color_lines[name[2]]}',label=f"{name[0]}-{name[2]}-lbp")
                ax4.plot(group["hidden nodes"],group["training time"],f'{act_lines[name[1]][name[0]]}o{color_lines[name[2]]}',label=f"{name[0]}-{name[2]}-lbp")
                ax5.plot(group["hidden nodes"],group["testing time"],f'{act_lines[name[1]][name[0]]}o{color_lines[name[2]]}',label=f"{name[0]}-{name[2]}-lbp")

            #Manually select best parameters based on the previous plots
            if (name[0] in ["Leaky ReLU","Swish"]) and (name[2] in ["hsv","rgb"]):
                ax3.plot(group["hidden nodes"],group["balanced accuracy"],f'{act_lines[name[1]][name[0]]}o{color_lines[name[2]]}',label=f"{name[0]}-{name[2]}{'-lbp' if name[1]==True else ''}")


        ax1.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncols=3) #legend showing name of each curve
        fig1.tight_layout()
        fig1.show()
        fig1.savefig(os.path.join(results_path,f"Accuracy"),bbox_inches="tight",pad_inches=1.0) #save response curve to file


        ax2.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncols=3) #legend showing name of each curve
        fig2.tight_layout()
        fig2.show()
        fig2.savefig(os.path.join(results_path,f"Accuracy_lbp"),bbox_inches="tight",pad_inches=1.0) #save response curve to file


        ax3.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncols=4) #legend showing name of each curve
        fig3.tight_layout()
        fig3.show()
        fig3.savefig(os.path.join(results_path,f"Accuracy_best"),bbox_inches="tight",pad_inches=1.0) #save response curve to file


        ax4.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncols=3) #legend showing name of each curve
        fig4.tight_layout()
        fig4.savefig(os.path.join(results_path,f"Training_time"),bbox_inches="tight",pad_inches=1.0) #save response curve to file


        ax5.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncols=3) #legend showing name of each curve
        fig5.tight_layout()
        fig5.savefig(os.path.join(results_path,f"Testing_time"),bbox_inches="tight",pad_inches=1.0) #save response curve to file


plt.waitforbuttonpress()
plt.clf()
plt.close()