#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import copy
import cv2, time, math, json
import numpy as np
import solver



input_shape = (28, 28, 1)
num_classes = 9
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.load_weights("digitRecognition.h5") 

video = cv2.VideoCapture(0)

x = 1

while True:
    check, frame = video.read()
    og = frame
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    maxA = -1
    maxC = None
    secA = -1
    secC = None
    
    for f in contours:
        if cv2.contourArea(f) > maxA:
            maxA = cv2.contourArea(f)
            maxC = f
        elif cv2.contourArea(f) > secA:
            secA = cv2.contourArea(f)
            secC = f
    
    
    big = [1]
    big[0] = secC 
    
      
    
    epsilon = 0.025*cv2.arcLength(secC, True)
    if secC is not None:
      poly = cv2.approxPolyDP(secC, epsilon, True)
   

    if len(poly) > 3:
                
        topleft =       min(secC, key=lambda x: x[0,0]+x[0,1])
        bottomright =   max(secC, key=lambda x: x[0,0]+x[0,1])
        topright =      max(secC, key=lambda x: x[0,0]-x[0,1])
        bottomleft =    min(secC, key=lambda x: x[0,0]-x[0,1])
        corners = (topleft, topright, bottomleft, bottomright)

        rect = cv2.minAreaRect(poly)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = np.float32([corners[0],corners[1],corners[2],corners[3]])
        dst_pts = np.float32([[0,0],[width,0],[0,height],[width,height]])

        per = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warp = cv2.warpPerspective(frame, per, (width, height))
        color_warp = cv2.warpPerspective(og, per, (width, height))
       
        dist = max(width, height)
        sq = round(dist/9)
        tiny = warp[0:sq,0:sq]
        squares = [None]*81
        a = 0
        height = warp.shape[0] // 9
        width = warp.shape[1] // 9
        offset_width = math.floor(width / 10)    
        offset_height = math.floor(height / 10)
        height = warp.shape[0] // 9
        width = warp.shape[1] // 9
        for y in range(0,9):
            for x in range(0,9):
                
                squares[a] = warp[height*y+offset_height:height*(y+1), width*x+offset_width:width*(x+1)-offset_width]    
                if squares[a] is not None:
                    if squares[a][:,0:3].any() is not 255:
                        squares[a] = warp[height*y+offset_height:height*(y+1)+offset_height, width*x+round(1.5*offset_width):width*(x+1)-offset_width]    
                    
                    if y >= 3 and x>= 3:   
                        height_rn = height*y+round(2*offset_height)
                        squares[a] = warp[height*y+round(2*offset_height):height*(y+1)+round(2*offset_height), width*x+round(2.5*offset_width):width*(x+1)-offset_width]              
                        if squares[a][0:3,:].any() is not 255:
                            squares[a] = warp[height_rn + 5:height*(y+1), width*x+offset_width:width*(x+1)-offset_width]    
                    
                    
                    ratio = 0.6      
                 
                    while np.sum(squares[a][0]) <= (1-ratio) * squares[a].shape[1] * 255:
                        squares[a] = squares[a][1:]
                    
                    while np.sum(squares[a][:,-1]) <= (1-ratio) * squares[a].shape[1] * 255:
                        squares[a] = np.delete(squares[a], -1, 1)
                  
                    while np.sum(squares[a][:,0]) <= (1-ratio) * squares[a].shape[0] * 255:
                        squares[a] = np.delete(squares[a], 0, 1)
                    
                    while np.sum(squares[a][-1]) <= (1-ratio) * squares[a].shape[0] * 255:
                        squares[a] = squares[a][:-1]    
                   
                    squares[a] = cv2.resize(squares[a], (28,28), interpolation = cv2.INTER_CUBIC)
                    center_width = squares[a].shape[1] // 2
                    center_height = squares[a].shape[0] // 2
                    x_start = center_height // 2
                    x_end = center_height // 2 + center_height
                    y_start = center_width // 2
                    y_end = center_width // 2 + center_width
                    center_region = squares[a][x_start:x_end, y_start:y_end]
                    if center_region.sum() >= center_width * center_height * 255 - 255:
                        squares[a] = None
                    else:
                        squares[a][0:4,0:4] = 255
                        image, contours, hierarchy = cv2.findContours(squares[a], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        maxA = -1
                        maxC = None
        
                        for f in contours:
                            if cv2.contourArea(f) > maxA:
                                maxA = cv2.contourArea(f)
                                maxC = f

                        epsilon = 0.025*cv2.arcLength(maxC, True)
                        poly = cv2.approxPolyDP(maxC, epsilon, True)
                        rect = cv2.minAreaRect(poly)
                        
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        pts = box
                     
                        
                        maxx = box[0][1] 
                        minx = box[2][1] 
                        maxy = box[0][0] 
                        miny = box[1][0] 
                       
                        
                       
                        croped = squares[a][miny:maxy+2, minx:maxx+2]

                       
                        if croped.shape[0] > 0 and croped.shape[1] > 0:
                            squares[a] = croped
                            squares[a] = cv2.resize(squares[a], (28,28), interpolation = cv2.INTER_CUBIC)

                        
                    a+=1            

       
        check, frame2 = video.read()

        contains = [None] * 81
        p_grid = [0] * 81
        for g in range(0,81):
            if squares[g] is not None:                
                img = squares[g] 
                img = img.astype(np.uint8)    
                img = img.reshape(-1, 28, 28, 1)
                img = img.astype('float32')
                img /= 255  
                prediction = model.predict([img])

                p_grid[g] = np.argmax(prediction[0]) + 1

   
         
    final_grid = np.array(p_grid)
    final_grid = np.reshape(final_grid,(9,9))
    open_grid = np.reshape(final_grid,(9,9))
    ##sol = solver.solve_sudoku(final_grid)  
    
 
    grid = np.array([[3,0,0,8,0,1,0,0,2],[2,0,1,0,3,0,6,0,4],[0,0,0,2,0,4,0,0,0],[8,0,9,0,0,0,1,0,6],[0,6,0,0,0,0,0,5,0],[7,0,2,0,0,0,4,0,9],[0,0,0,5,0,9,0,0,0],[9,0,4,0,8,0,7,0,5],[6,0,0,1,0,7,0,0,3]])
    sol = np.array([[3,0,0,8,0,1,0,0,2],[2,0,1,0,3,0,6,0,4],[0,0,0,2,0,4,0,0,0],[8,0,9,0,0,0,1,0,6],[0,6,0,0,0,0,0,5,0],[7,0,2,0,0,0,4,0,9],[0,0,0,5,0,9,0,0,0],[9,0,4,0,8,0,7,0,5],[6,0,0,1,0,7,0,0,3]])
    solver.solve_sudoku(final_grid)
    solver.solve_sudoku(sol)
    
    
    
    if sol is not None:
        SIZE = 9
        width = color_warp.shape[1] // 9
        height = color_warp.shape[0] // 9
        for i in range(SIZE):
            for j in range(SIZE):
                if grid[i][j] == 0: 
                    text = str(sol[i][j])
                    off_set_x = width // 15
                    off_set_y = height // 15
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=1)
                    marginX = math.floor(width / 7)
                    marginY = math.floor(height / 7)
                    font_scale = 0.6 * min(width, height) / max(text_height, text_width)
                    text_height *= font_scale
                    text_width *= font_scale
                    bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + off_set_x
                    bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + off_set_y
                    color_warp = cv2.putText(color_warp, text, (bottom_left_corner_x, bottom_left_corner_y),font, font_scale, (255,0,0), thickness=3, lineType=cv2.LINE_AA)    
                   
       
    final = cv2.warpPerspective(color_warp, per, (og.shape[1], og.shape[0])
                                        , flags=cv2.WARP_INVERSE_MAP)
    final = np.where(final.sum(axis=-1,keepdims=True)!=0, final, og)
    
    cv2.imshow('capture',final)
            
    key = cv2.waitKey(1)
    if key == ord('x'):
        break
        
video.release()
cv2.destroyAllWindows



# In[ ]:




