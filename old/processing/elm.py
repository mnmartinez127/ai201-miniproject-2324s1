#Extreme Learning Machine implementation for the AI 201 miniproject
import numpy as np
#Use as a backup
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.utils.multiclass import unique_labels, type_of_target
#base class for Extreme Learning Machine for classification
class ELMClassifier:

    #ACTIVATION FUNCTIONS
    #define the hyperbolic tangent function with a=1.716 and b=2/3
    def tanh_f(self,v,a=1.716,b=(2.0/3.0)): #phi(v) = a*tanh(b*v)
        return a*np.tanh(b*v)

    #define the logistic sigmoid function with a=2.0
    def log_f(self,v,a=2.0) : #phi(v) = 1/(1+e^(-a*v)) = (e^(a*v))/((e^(a*v))+1)
        return np.where(v>=0, 1/(1+np.exp(-a*v)), np.exp(a*v)/(np.exp(a*v)+1))

    #define the swish function with b=1.0
    def swish_f(self,v,b=1.0) : #phi(v) = v/(1+e^(-a*v)) = v(e^(a*v))/((e^(a*v))+1)
        return v*self.log_f(v,b)

    #define the leaky ReLU function with gradient parameter a=0.01
    def lrelu_f(self,v,a=0.01): #phi(v) = v if v > 0, a*v if v < 0
        return np.where(v>0, v,a*v)

    #define the ELU function with gradient parameter a=0.01
    def elu_f(self,v,a=1.67326,l=1.0507): #phi(v) = lv if v >= 0, la(e^v - 1) if v < 0
        return np.where(v>=0, l*v,l*a*(np.exp(v)-1))

    def linear_f(self,v): #phi(v) = v
        return v

    def __init__(self, n_neurons=None, ufunc="tanh", pairwise_metric=None, random_state=None):
        self.randomizer = np.random.default_rng(random_state)
        self.pairwise_metric = pairwise_metric
        self.set_act_func("linear") #initialize activation function to linear activation
        self.init_hidden(n_neurons,ufunc) #initialize ELM network
    
    def init_hidden(self,num_hidden=None,act_func=None):
        if num_hidden: self.num_hidden=num_hidden #update the number of hidden layer nodes
        if act_func: self.set_act_func(act_func) #update the activation function

    def set_act_func(self,act_func):
        match act_func:
            case "tanh":
                self.act_func = self.tanh_f
            case "log":
                self.act_func = self.log_f
            case "lrelu":
                self.act_func = self.lrelu_f
            case "elu":
                self.act_func = self.elu_f
            case "swish":
                self.act_func = self.swish_f
            case "linear":
                self.act_func = self.linear_f
            case _:
                pass #do not update activation function
        if callable(act_func): #check if function
            self.act_func = act_func


    def fit(self,X,y=None):
        self.num_features,self.num_classes = len(X.T),len(y) #update number of input and output nodes

        if not hasattr(self, 'label_binarizer'):
            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(y)
        
        y_numeric = self.label_binarizer.transform(y)

        self.hidden_weights = self.randomizer.normal(size=(self.num_features,self.num_hidden)) #hidden layer weights
        self.output_weights = np.zeros((self.num_features,self.num_classes)) #hidden layer weights
        self.bias = self.randomizer.normal(size=(self.num_hidden)) #bias weights
        if not hasattr(self,"num_hidden") or self.num_hidden == None:
            print("Error: number of hidden nodes is not specified.")
            return
        #forward step
        if self.pairwise_metric == None:
            v = (X@self.hidden_weights)+self.bias #input to hidden nodes
        else:
            v = pairwise_distances(X,self.num_hidden,n_jobs=None,metric=self.pairwise_metric) #use radial basis function
        H = self.act_func(v) #output of hidden nodes
        y_numeric = self.label_binarizer.transform(y)
        self.output_weights = np.linalg.pinv(H)@y_numeric #use Moore-Penrose inverse
        return self

    def predict(self,X):
        if not (hasattr(self,"output_weights") and hasattr(self,"hidden_weights")) or self.hidden_weights is None or self.output_weights is None:
            print("Error: network is not trained!")
            return
        if (1 if X.ndim < 2 else X.shape[1]) != self.hidden_weights.shape[0]:
            print(f"Error: number of features {(1 if X.ndim < 2 else X.shape[1])} is less than expected number {self.hidden_weights.shape[0]}")
            return
        if self.pairwise_metric == None:
            v = (X@self.hidden_weights)+self.bias #input to hidden nodes
        else:
            v = pairwise_distances(X,self.num_hidden,n_jobs=None,metric=self.pairwise_metric) #use radial basis function
        H = self.act_func(v) #output of hidden nodes
        y = H@self.output_weights #compute the outputs from trained weights
        y = self.label_binarizer.inverse_transform(y)
        return y