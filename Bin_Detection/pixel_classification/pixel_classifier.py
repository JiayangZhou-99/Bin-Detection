'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
class LogisticRegression():
    def __init__(self):
        '''
    	    Initilize your classifier with any parameters and attributes you need
        '''  
        self.w = np.random.rand(1, 4)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    
    def train(self,X,y,iteration = 100,learning_rate = 0.01,lamda = 0.1):
        
        bias  = np.ones((X.shape[0],1))
        X_new = np.concatenate((bias,X),axis = 1)
        sample_num = X.shape[0]
        y = np.array([y]).reshape((-1,1))
        for i in range (0,2000):
            Y_pre    = np.dot(X_new,self.w.T)
            Y_pre    = self.sigmoid(Y_pre)
            gradient = (-np.dot((y - Y_pre).T,X_new) + lamda*np.sum(self.w))/sample_num
            self.w  -=  learning_rate * gradient
    def getW(self):
        return self.w
	
    def classify(self,X):
        '''
    	    Classify a set of pixels into red, green, or blue
    	    
    	    Inputs:
    	      X: n x 3 matrix of RGB values
    	    Outputs:
    	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        # Just a random classifier for now
        # Replace this with your own approach 
        
        bias  = np.ones((X.shape[0],1))
        X_new = np.concatenate((bias,X),axis = 1)
        y_pred = np.dot(X_new,self.w.T)
        y_pred = self.sigmoid(y_pred)
        # YOUR CODE BEFORE THIS LINE
        return y_pred
    
    

class PixelClassifier():
    def __init__(self):
        '''
    	    Initilize your classifier with any parameters and attributes you need
        '''  
        self.model1 = LogisticRegression() ##distinguish 1 and 2&3
        self.model2 = LogisticRegression() ##distinguish 2 and 1&3
        self.model3 = LogisticRegression() ##distinguish 3 and 1&2
    
    def train(self,X,y,learning_rate = 0.1,lamda = 0.1):
        
        ##first split the data and put them into seperate training data
        
        for i in range (0,y.shape[0]):
            if y[i] == 2:
                two_start = i
                break
        for i in range (0,y.shape[0]):
            if y[i] == 3:
                three_start = i
                break
            
            
        one_y = np.zeros((y.shape[0],1))
        one_y[:two_start] = 1
        
        two_y = np.zeros((y.shape[0],1))
        two_y[two_start:three_start] = 1
        
        thr_y =np.zeros((y.shape[0],1))
        thr_y[three_start:] = 1
        
        self.model1.train(X,one_y,learning_rate,lamda)
        self.model2.train(X,two_y,learning_rate,lamda)
        self.model3.train(X,thr_y,learning_rate,lamda)
        return np.concatenate((self.model1.getW(),self.model2.getW(),self.model3.getW()),axis= 0)
	
    def classify(self,X):
        '''
    	    Classify a set of pixels into red, green, or blue
    	    
    	    Inputs:
    	      X: n x 3 matrix of RGB values
    	    Outputs:
    	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        # Just a random classifier for now
        # Replace this with your own approach 
        # one = self.model1.classify(X)
        # two = self.model2.classify(X)
        # thr = self.model3.classify(X)
        # all_data = np.concatenate((one,two,thr),axis = 1)
        weight = np.array([[-1.17227135,  6.38322718, -3.30960737, -3.19249442],
                            [-0.95560608, -3.63335187,  6.13819129, -3.20317297],
                            [-0.97357361, -3.56814519, -3.2084153,   6.03092903]])
        bias  = np.ones((X.shape[0],1))
        X_new = np.concatenate((bias,X),axis = 1)
        all_data = np.dot(weight,X_new.T)
        y_pred   = np.argmax(all_data, axis=0)
        y_pred  += 1
        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return y_pred
        

