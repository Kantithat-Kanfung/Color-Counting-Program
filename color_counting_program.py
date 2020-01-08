# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:45:38 2018

@author: lieka
"""

# -- import modules --
import numpy as np
import cv2
import Person
import time

# -- Entry and exit counters --
cnt_red = 0
cnt_blue = 0
cnt_yellow = 0
cnt_green = 0
cnt_purple = 0
cnt_cyan = 0

# -- capturing video through webcam --
cap = cv2.VideoCapture(0)

# -- Print the capture properties to console --
for i in range(15):
    print (i, cap.get(i))

w = cap.get(3)
h = cap.get(4)
frameArea = h * w
areaTH = frameArea/250
print ('Area Threshold', areaTH)

line_middle = int(3*(h/6))

up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))

print ("Red line y:",str(line_middle))

line_middle_color = (15,255,255)

pt1 =  [0, line_middle];
pt2 =  [w, line_middle];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))

pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))

pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

# -- Create the background substractor --
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

# -- Structuring elements for morphographic filters --
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

# -- Variables --
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while(cap.isOpened()):
            
    # -- Read an image of the video source --
    ret, frame = cap.read()
    
    # -- age every person one frame --
    for i in persons:
        i.age_one() 
        
    # -- Apply background subtraction --
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # -- Binarization to eliminate shadows -- (Gray)
    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        
        red_lower = np.array([125, 95, 145], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        
        red = cv2.inRange(hsv, red_lower, red_upper)
        red = cv2.dilate(red, kernelOp2)
        
        # -- definding the range of blue color --
        blue_lower = np.array([110, 150, 80], np.uint8)
        blue_upper = np.array([130, 255, 255], np.uint8)
        
        blue = cv2.inRange(hsv, blue_lower, blue_upper)
        blue = cv2.dilate(blue, kernelOp2)
        
        # -- definding the range of yellow color --
        yellow_lower = np.array([30, 100, 110], np.uint8)
        yellow_upper = np.array([40, 255, 255], np.uint8)
        
        yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow = cv2.dilate(yellow, kernelOp2)
        
        # -- definding the range of green color --
        green_lower = np.array([55, 115, 45], np.uint8)
        green_upper = np.array([80, 255, 255], np.uint8)
        
        green = cv2.inRange(hsv, green_lower, green_upper)
        green = cv2.dilate(green, kernelOp2)
         
        # -- definding the range of purple color --
        purple_lower = np.array([130, 90, 20], np.uint8)
        purple_upper = np.array([160, 255, 255], np.uint8)
        
        purple = cv2.inRange(hsv, purple_lower, purple_upper)
        purple = cv2.dilate(purple, kernelOp2)
        
        # -- definding the range of cyan color --
        cyan_lower = np.array([90, 60, 110], np.uint8)
        cyan_upper = np.array([100, 255, 255], np.uint8)
        
        cyan = cv2.inRange(hsv, cyan_lower, cyan_upper)
        cyan = cv2.dilate(cyan, kernelOp2)
       
    except:
        print('EOF')
        print('DOWN: ', cnt_red)
        print('DOWN: ', cnt_blue)
        print('DOWN: ', cnt_yellow)
        print('DOWN: ', cnt_green)
        print('DOWN: ', cnt_purple)
        print('DOWN: ', cnt_cyan)
        
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # -- Tracking --    
            
            # -- Need to add conditions for multipersons, outputs and screen inputs ---
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        
                        # -- The object is near one that is already detected before --
                        new = False
                       
                        # -- actualiza coordinates in object and resets age --
                        i.updateCoords(cx,cy)   
                        
                        if i.going_DOWN(line_middle) == True:
                            cnt_red += 1;
                            print ("ID:",i.getId(),'Red groups down at',time.strftime("%c"))
                        break
                    
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        
                    if i.timedOut():
                        # -- Remove I from the persons list --
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # -- Release the memory of I --
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 7, (0,0,255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # -- Tracking --    
            
            # -- Need to add conditions for multipersons, outputs and screen inputs ---
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        
                        # -- The object is near one that is already detected before --
                        new = False
                       
                        # -- actualiza coordinates in object and resets age --
                        i.updateCoords(cx,cy)   
                        
                        if i.going_DOWN(line_middle) == True:
                            cnt_blue += 1;
                            print ("ID:",i.getId(),'Blue groups down at',time.strftime("%c"))
                        break
                    
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        
                    if i.timedOut():
                        # -- Remove I from the persons list --
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # -- Release the memory of I --
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 7, (255, 0, 0), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
    
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(yellow,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # -- Tracking --    
            
            # -- Need to add conditions for multipersons, outputs and screen inputs ---
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        
                        # -- The object is near one that is already detected before --
                        new = False
                       
                        # -- actualiza coordinates in object and resets age --
                        i.updateCoords(cx,cy)   
                        
                        if i.going_DOWN(line_middle) == True:
                            cnt_yellow += 1;
                            print ("ID:",i.getId(),'Yellow groups down at',time.strftime("%c"))
                        break
                    
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        
                    if i.timedOut():
                        # -- Remove I from the persons list --
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # -- Release the memory of I --
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 7, (0, 255, 255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 255),2)
            
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(green,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # -- Tracking --    
            
            # -- Need to add conditions for multipersons, outputs and screen inputs ---
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        
                        # -- The object is near one that is already detected before --
                        new = False
                       
                        # -- actualiza coordinates in object and resets age --
                        i.updateCoords(cx,cy)   
                        
                        if i.going_DOWN(line_middle) == True:
                            cnt_green += 1;
                            print ("ID:",i.getId(),'Green groups down at',time.strftime("%c"))
                        break
                    
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        
                    if i.timedOut():
                        # -- Remove I from the persons list --
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # -- Release the memory of I --
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 7, (0, 255, 0), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)
            
        # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(green,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # -- Tracking --    
            
            # -- Need to add conditions for multipersons, outputs and screen inputs ---
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        
                        # -- The object is near one that is already detected before --
                        new = False
                       
                        # -- actualiza coordinates in object and resets age --
                        i.updateCoords(cx,cy)   
                        
                        if i.going_DOWN(line_middle) == True:
                            cnt_green += 1;
                            print ("ID:",i.getId(),'Green groups down at',time.strftime("%c"))
                        break
                    
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        
                    if i.timedOut():
                        # -- Remove I from the persons list --
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # -- Release the memory of I --
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 6, (0, 255, 0), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)
            
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(purple,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # -- Tracking --    
            
            # -- Need to add conditions for multipersons, outputs and screen inputs ---
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        
                        # -- The object is near one that is already detected before --
                        new = False
                       
                        # -- actualiza coordinates in object and resets age --
                        i.updateCoords(cx,cy)   
                        
                        if i.going_DOWN(line_middle) == True:
                            cnt_purple += 1;
                            print ("ID:",i.getId(),'Purple groups down at',time.strftime("%c"))
                        break
                    
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        
                    if i.timedOut():
                        # -- Remove I from the persons list --
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # -- Release the memory of I --
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 6, (170, 0, 127), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(170, 0, 127),2)

    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(cyan,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # -- Tracking --    
            
            # -- Need to add conditions for multipersons, outputs and screen inputs ---
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        
                        # -- The object is near one that is already detected before --
                        new = False
                       
                        # -- actualiza coordinates in object and resets age --
                        i.updateCoords(cx,cy)   
                        
                        if i.going_DOWN(line_middle) == True:
                            cnt_cyan += 1;
                            print ("ID:",i.getId(),'Cyan groups down at',time.strftime("%c"))
                        break
                    
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        
                    if i.timedOut():
                        # -- Remove I from the persons list --
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # -- Release the memory of I --
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 6, (255, 165, 0), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 165, 0),2)
    
    # -- Drawing paths --
    
    for i in persons:
        if len(i.getTracks()) >= 2:
            cv2.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.6,i.getRGB(),2,cv2.LINE_AA)
    
    name_comp = 'BANGKOK RANCH PLC.' 
    red_down = 'RED-GROUP : ' + str(cnt_red) 
    blue_down = 'BLUE-GROUP : ' + str(cnt_blue) 
    yellow_down = 'YELLOW-GROUP : ' + str(cnt_yellow) 
    green_down = 'GREEN-GROUP : ' + str(cnt_green)
    purple_down = 'PURPLE-GROUP : ' + str(cnt_purple) 
    cyan_down = 'CYAN-GROUP : ' + str(cnt_cyan) 
    frame = cv2.polylines(frame,[pts_L1],False,line_middle_color,thickness=2)
    
    frame = cv2.polylines(frame,[pts_L3],False,(255, 255, 255),thickness=1)
    frame = cv2.polylines(frame,[pts_L4],False,(255, 255, 255),thickness=1)
    
    cv2.putText(frame, name_comp ,(170,30),cv2.FONT_HERSHEY_TRIPLEX,0.9,(10, 80, 0),2)
    cv2.putText(frame, red_down ,(30,60),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),1)
    cv2.putText(frame, blue_down ,(30,80),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255, 0, 0),1)
    cv2.putText(frame, yellow_down ,(230,60),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0, 255, 255),1)
    cv2.putText(frame, green_down ,(230,80),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0, 255, 0),1)
    cv2.putText(frame, purple_down ,(430,60),cv2.FONT_HERSHEY_TRIPLEX,0.5,(170, 0, 127),1)
    cv2.putText(frame, cyan_down ,(430,80),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255, 165, 0),1)
    
    cv2.imshow('Frame', frame)
       
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()