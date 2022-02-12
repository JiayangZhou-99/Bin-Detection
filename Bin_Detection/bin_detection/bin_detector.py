'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''
import os, cv2
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt; plt.ion()

class BinDetector():
    def __init__(self):
        '''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
        folder_path = os.path.dirname(os.path.abspath(__file__)) 
        model_params_file = os.path.join(folder_path, 'final_weights.npy')
        l = np.load(model_params_file)
        self.w = np.array(l)
        # self.w= np.array([[-19.83193938, -15.1979202,   18.22414677,  -4.89965558],
        #                     [0.96083018,  -0.97452232,   0.82823063,  -0.62109639],
        #                     [  0.59994694,  -7.98056373,  -2.70022244,   5.29974882],
        #                     [  0.94091389  ,10.63240337,  -2.97798333,  -4.10705371],
        #                     [  0.16690198  ,-8.01426128, -13.40178862, -12.73443539]])
  
    def segment_image(self, img):
	     
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        mask_img = np.zeros(img.shape)
        
        for i in range (0,img.shape[0]):
            for j in range (0,img.shape[1]):
                new_data = np.append(np.array([1]),img[i,j])
                out      = np.dot(self.w,new_data.T)
                color    = np.argmax(out,axis = 0)
                if color == 0:
                    #blue
                    mask_img[i,j] = np.array([0,0,1])
                if color == 1:
                    #Dgreen
                    mask_img[i,j] = np.array([0,100/255,0])
                if color == 2:
                    #Brown
                    mask_img[i,j] = np.array([184/255,134/255,11/255])
                if color == 3:
                    mask_img[i,j] = np.array([0,0,0])
                if color == 4:
                    mask_img[i,j] = np.array([192/255,192/255,192/255])
        return mask_img
    def train():
        train()
    def get_bounding_boxes(self, img):
        '''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (9,6))  # 矩形结构
        erosion = cv2.erode(img, kernel, iterations=1)
        dilated = cv2.dilate(erosion, kernel, 2)
        plt.imshow(dilated)
        plt.show()
        label_img = label(dilated, connectivity=dilated.ndim)
        props = regionprops(label_img)
        boxes = []
        for prop in props:
            minr, minc,_, maxr, maxc,_ = prop.bbox
            if (maxc-minc)*(maxr-minr) >= 10000:
                similarity = (maxr - minr)/(maxc - minc)
                saturate = 0
                for i in range (minc,maxc):
                    for j in range(minr,maxr):
                        if img[j,i].all() == np.array([0,0,1]).all():
                            saturate += 1
                saturate /= (maxc-minc)*(maxr-minr)
                if 1 < similarity < 2 and saturate > 0.8:
                    boxes.append([minc, minr, maxc, maxr])
        return boxes


