import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

izoBackground = cv2.imread("Resources/Background.png")
izoGameOver = cv2.imread("Resources/gameOver.png")
izoBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
izoBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
izoBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

detector = HandDetector(detectionCon=0.8, maxHands=2)

ballPos = [100, 100]
naprOX = 15
napr0Y = 15
gameOver = False
shet = [0, 0]


while True:
    _, izo = cap.read()
    izo = cv2.flip(izo, 1)
    izoRaw = izo.copy()
    
    hands, izo = detector.findHands(izo, flipType=False)

    izo = cv2.addWeighted(izo, 0.2, izoBackground, 0.8, 0)

    if hands:
        for hand in hands: 
            x, y, w, h = hand['bbox'] 
            h1, w1, _ = izoBat1.shape 
            y1 = y - h1//2 
            y1 = np.clip(y1, 20, 415) 

            if hand['type'] == "Left": 
                izo = cvzone.overlayPNG(izo, izoBat1, (59, y1)) 
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1: 
                    naprOX = -naprOX 
                    ballPos[0] += 30
                    shet[0] += 1 

            if hand['type'] == "Right": 
                izo = cvzone.overlayPNG(izo, izoBat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 + w1 and y1 < ballPos[1] < y1 + h1:
                    naprOX = -naprOX 
                    ballPos[0] -= 30
                    shet[1] += 1

    
    if ballPos[0] < 40  or ballPos[0] > 1200: 
        gameOver = True
    
    if gameOver:
        izo = izoGameOver

   
    else:
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            napr0Y = -napr0Y

        ballPos[0] += naprOX
        ballPos[1] += napr0Y

       
        izo = cvzone.overlayPNG(izo, izoBall, ballPos) 

        
        cv2.putText(izo, str(shet[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(izo, str(shet[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    
    izo[580:700, 20:233] = cv2.resize(izoRaw, (213, 120)) 

    cv2.imshow("Image", izo)
    key = cv2.waitKey(1)


    if key == ord("r"):
        ballPos = [100, 100]
        naprOX = 15
        napr0Y = 15
        gameOver = False
        shet = [0, 0]
        izoGameOver = cv2.imread("Resources/gameOver.png")