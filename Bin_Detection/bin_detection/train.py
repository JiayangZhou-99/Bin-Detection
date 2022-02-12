#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:58:18 2022

@author: zhoujiayang
"""

from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2
matplotlib.use('Qt5Agg')

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
        self.model4 = LogisticRegression() ##distinguish 4 and 1&2
        self.model5 = LogisticRegression() ##distinguish 4 and 1&2
    
    def getW(self):
        
        ##get the final weight matrix...
        return np.concatenate((self.model1.getW(),self.model2.getW(),self.model3.getW(),self.model4.getW(),self.model5.getW()))
    
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
        for i in range (0,y.shape[0]):
            if y[i] == 4:
                four_start = i
                break
        for i in range (0,y.shape[0]):
            if y[i] == 5:
                five_start = i
                break
            
        one_y = np.zeros((y.shape[0],1))
        one_y[:two_start] = 1
        
        two_y = np.zeros((y.shape[0],1))
        two_y[two_start:three_start] = 1
        
        thr_y =np.zeros((y.shape[0],1))
        thr_y[three_start:four_start] = 1
        
        fou_y =np.zeros((y.shape[0],1))
        fou_y[four_start:five_start] = 1
        
        fiv_y =np.zeros((y.shape[0],1))
        fou_y[five_start:] = 1
        
        self.model1.train(X,one_y,learning_rate,lamda)
        self.model2.train(X,two_y,learning_rate,lamda)
        self.model3.train(X,thr_y,learning_rate,lamda)
        self.model4.train(X,fou_y,learning_rate,lamda)
        self.model5.train(X,fiv_y,learning_rate,lamda)
        
        
	
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
        one = self.model1.classify(X)
        two = self.model2.classify(X)
        thr = self.model3.classify(X)
        fou = self.model4.classify(X)
        fiv = self.model5.classify(X)
        
        
        all_data = np.concatenate((one,two,thr,fou,fiv),axis = 1)
        y_pred   = np.argmax(all_data, axis=1)
        y_pred  += 1
        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return y_pred

def read_pixels(folder, verbose = False):
  '''
    Reads 3-D pixel value of the top left corner of each image in folder
    and returns an n x 3 matrix X containing the pixel values 
  '''  
  n = len(next(os.walk(folder))[2]) # number of files
  X = np.empty([n, 3])
  i = 0
  
  if verbose:
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))
  
  for filename in os.listdir(folder):  
    # read image
    # img = plt.imread(os.path.join(folder,filename), 0)
    img = cv2.imread(os.path.join(folder,filename))
    # convert from BGR (opencv convention) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # store pixel rgb value
    X[i] = img[0,0].astype(np.float64)/255
    i += 1
    
    # display
    if verbose:
      h.set_data(img)
      ax.set_title(filename)
      fig.canvas.flush_events()
      plt.show()

  return X

def get_maskAndImag(PID):
    # read the first training image
    folder = 'data/training'
    filename = '00'+ PID + '.jpg'  
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display the image and use roipoly for labeling
    fig, ax = plt.subplots()
    ax.imshow(img)
    my_roi = RoiPoly(fig=fig, ax=ax, color='r')
    
    # get the image mask
    mask = my_roi.get_mask(img)
    print(mask)
    # display the labeled region and the image mask
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
    
    ax1.imshow(img)
    ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
    ax2.imshow(mask)
    
    plt.show(block=True)
    return img,mask


if __name__ == '__main__':
    
    img_blue,mask_blue = get_maskAndImag('01')
    np.save("image_blue.npy",img_blue)
    np.save("mask_blue.npy",mask_blue)
    
    img_Dgreen,mask_Dgreen = get_maskAndImag('02')
    np.save("img_Dgreen.npy",img_Dgreen)
    np.save("mask_Dgreen.npy",mask_Dgreen)
    
    img_Brown,mask_Brown = get_maskAndImag('12')
    np.save("image_Brown",img_Brown)
    np.save("mask_Brown",mask_Brown)
    
    img_Black,mask_Black = get_maskAndImag('04')
    
    img_gray ,mask_gray = get_maskAndImag('22')
    
    
    img_blue   = cv2.cvtColor(img_blue,cv2.COLOR_RGB2HSV)
    img_Dgreen = cv2.cvtColor(img_Dgreen,cv2.COLOR_RGB2HSV)
    img_Brown  = cv2.cvtColor(img_Brown,cv2.COLOR_RGB2HSV)
    img_Black  = cv2.cvtColor(img_Black,cv2.COLOR_RGB2HSV)
    img_gray  = cv2.cvtColor(img_gray,cv2.COLOR_RGB2HSV)
    
    ##first enumerate the original image to get the training data points
    X1 = [] #1 represents blue
    X2 = [] #2 represents Dgreen
    X3 = [] #3 represents Brown
    X4 = [] #4 represents Black
    X5 = [] #5 represents gray
    
    for i in range (0,img_blue.shape[0]):
        for j in range (0,img_blue.shape[1]):
            if mask_blue[i,j] == True:
                X1.append(img_blue[i,j])
                
    for i in range (0,img_Dgreen.shape[0]):
        for j in range (0,img_Dgreen.shape[1]):
            if mask_Dgreen[i,j] == True:
                X2.append(img_Dgreen[i,j])
        
    for i in range (0,img_Brown.shape[0]):
        for j in range (0,img_Brown.shape[1]):
            if mask_Brown[i,j] == True:
                X3.append(img_Brown[i,j])

                
    for i in range (0,img_Black.shape[0]):
        for j in range (0,img_Black.shape[1]):
            if mask_Black[i,j] == True:
                X4.append(img_Black[i,j])
                
    for i in range (0,img_gray.shape[0]):
        for j in range (0,img_gray.shape[1]):
            if mask_gray[i,j] == True:
                X5.append(img_gray[i,j])
                
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    X5 = np.array(X5)
    
    myPixelClassifier = PixelClassifier()
    y1, y2, y3,y4,y5 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3), np.full(X4.shape[0],4),np.full(X5.shape[0],5)    
    X, y = np.concatenate((X1,X2,X3,X4,X5)), np.concatenate((y1,y2,y3,y4,y5))
    myPixelClassifier.train(X,y)
    res = myPixelClassifier.getW()
    np.save("final_weights.npy",res)
    print(res)
    