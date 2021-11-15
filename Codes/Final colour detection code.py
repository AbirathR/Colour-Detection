
import numpy as np
import cv2  
cap = cv2.VideoCapture(0)  
while(1):         
    _, imageFrame = cap.read()     
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)      
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)     
    green_lower = np.array([0, 91, 0], np.uint8)
    green_upper = np.array([179, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)      
    blue_lower = np.array([110,50,50], np.uint8)
    blue_upper = np.array([130,255,255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    orange_lower = np.array([5, 50, 50], np.uint8)
    orange_upper = np.array([15, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)  
    kernal = np.ones((5, 5), "uint8")          
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,mask = red_mask)        
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,mask = green_mask)          
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,mask = blue_mask)
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,mask = yellow_mask)
    orange_mask = cv2.dilate(orange_mask, kernal)
    res_orange = cv2.bitwise_and(imageFrame, imageFrame,mask = orange_mask) 
    contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 3000 and area<50000):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h), (0, 0, 255), 2)             
            cv2.putText(imageFrame, "Red Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255))
            print("1")
    contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 15000  and area<50000):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 255, 0), 2)  
            cv2.putText(imageFrame, "Green Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            print("2")
    contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 5000  and area<50000):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(255, 0, 0), 2)  
            cv2.putText(imageFrame, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 0, 0))
            print("3")
    contours, hierarchy = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 3000  and area<50000):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 255, 255), 2) 
            cv2.putText(imageFrame, "Yellow Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))
            print("4")
    contours, hierarchy = cv2.findContours(orange_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 7000 and area<50000):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h), (3, 99, 234), 2)             
            cv2.putText(imageFrame, "Orange Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(3,99, 234))
            print("5")
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
