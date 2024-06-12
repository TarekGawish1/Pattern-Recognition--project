import face_recognition
import cv2

# Function to recognize faces in an image
def recognize_image(img):
    # Load the known images and encode them
    known_images = []
    known_names = []

    known_images.append(face_recognition.load_image_file("Known\image1.jpg"))
    known_names.append("Elon Musk")

    known_images.append(face_recognition.load_image_file("Known\image3.jpeg"))
    known_names.append("Bill Gates")
    
    known_images.append(face_recognition.load_image_file("Known\image2.jpg"))
    known_names.append("Jeff Bezos")
    
    
    known_face_encodings = []
    for image in known_images:
        known_face_encodings.append(face_recognition.face_encodings(image)[0])

    # Encode the faces in the unknown image
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # Loop over each face in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding to the known face encodings to see if they match
        name = "Unknown Person"
        color = (0,0 ,255)  # red
        for i, known_face_encoding in enumerate(known_face_encodings):
            results = face_recognition.compare_faces([known_face_encoding], face_encoding)
            if results[0]:
                name = known_names[i]
                color = (0, 255, 0)  # green
                break
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return img

# Perform face recognition using the webcam
def recognize_from_camera():
    # Open the video capture device
    cap = cv2.VideoCapture(0)

    # Loop until the user quits
    while True:
        # Read a frame from the video capture device
        ret, frame = cap.read()

        # Recognize faces in the frame
        img = recognize_image(frame)

        # Display the resulting image
        cv2.imshow('Video Feed', img)

        # Check if the user wants to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    choice = input("Enter 'i' to recognize from image or 'c' to recognize from camera: ")
    if choice == 'i':
        img_path = input("Enter path to image: ")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (800, 500))
        img = recognize_image(img)
        cv2.imshow('Image Recognition', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif choice == 'c':
        recognize_from_camera()
    else:
        print("Invalid choice")

if __name__ == '__main__':
    main()
