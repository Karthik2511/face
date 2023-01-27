import cv2
import datetime

# Load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the font to be used on the output image
font = cv2.FONT_HERSHEY_SIMPLEX

# Load the trained model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")
with open("labels.txt", "r") as f:
    labels = {int(line.strip()): name for line,
              name in enumerate(f.readlines())}

while True:
    # Read the frame
    _, img = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate over each face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Recognize the face
        id, confidence = recognizer.predict(roi_gray)

        # Check the confidence level
        if confidence < 50:
            # Get the name of the person
            name = labels[id]

            # Log the information
            now = datetime.datetime.now()
            log = f"{now} - Recognized: {name} (confidence: {confidence})"
            print(log)
            with open("log.txt", "a") as f:
                f.write(log + "\n")

            # Put the name on the image
            cv2.putText(img, name, (x, y-10), font, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

    # Show the output
    cv2.imshow("img", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
