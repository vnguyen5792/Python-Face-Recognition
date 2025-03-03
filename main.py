import cv2 as cv

face_cascade = cv.CascadeClassifier('XMLClassifier/haarcascade_frontalface_default.xml')

# capture frames from a camera
cap = cv.VideoCapture(0)

#parameters for text
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2
color = (255, 255, 0)

#text size for centering
textSize, _ = cv.getTextSize('Detected Face', font, fontScale, thickness)
textWidth, _ = textSize

# loop runs if capturing has been initialized.
while True: 

    # reads frames from a camera
    ret, img = cap.read() 

    # convert to gray scale of each frames
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 

        #centers text with the center of the rectangle
        if(textWidth > w):
            origin = (x - abs((w//2 - textWidth//2)), y - 10)
        else:
            origin = (x + abs((w//2 - textWidth//2)), y - 10)

        cv.putText(img, 'Detected Face', origin, cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 0), 2)

    # Display an image in a window
    cv.flip(img, 1)
    cv.imshow('img',img)

    # Wait for Esc key to stop
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv.destroyAllWindows()