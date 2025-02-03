import cv2
import time
import imutils

# Initialize webcam (try index 0 or 1 if 2 doesn't work)
cam = cv2.VideoCapture(0)  
time.sleep(1)

firstFrame = None
area = 500  # Reduced area threshold for better detection

frame_count = 0  # To update the first frame periodically

while True:
    ret, img = cam.read()

    # Ensure the frame was successfully captured
    if not ret or img is None:
        print("Failed to capture image. Check your camera index or connection.")
        break

    text = "Normal"
    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    # Update the first frame every 100 frames to adapt to new background conditions
    if firstFrame is None or frame_count % 100 == 0:
        firstFrame = gaussianImg
        print("Updated first frame")
        frame_count = 0  # Reset counter

    frame_count += 1  # Increment frame count

    # Compute difference between first frame and current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Debugging: Print number of contours found
    print(f"Contours detected: {len(cnts)}")

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object Detected"

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Camera Reading", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release camera and close all OpenCV windows AFTER loop ends
cam.release()
cv2.destroyAllWindows()



    
    
    
    
    
