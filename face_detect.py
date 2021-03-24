#Import OpenCV
import cv2 
# Use Pillow library to add image resize in next iteration.
# from PIL import image
# im = Image.open('cat.jpeg')

# Create arguments for the image and classifier we want to use.
imagePath = 'cat_two.jpeg'
cascPath = "haarcascade_frontalcatface.xml" 

# Pass in argument from line 9 to create the haar cascade.
faceCascade = cv2.CascadeClassifier(cascPath) 

# Pass in imagePath argument from line 8 to read the image and then to convert to grayscale.
image = cv2.imread(imagePath)  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE. caused error when code ran. Explore later. 
)

print("Found {0} faces!".format(len(faces)))

# Faces will return a list of coordinates. Create a for in loop to iterate through them and then draw a rectangle around the faces
for (x, y, w, h) in faces:  
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Allows user to see the image and then wait indefinitely to exit with any key press. 
cv2.imshow("Faces found", image)  
cv2.waitKey(0) 
