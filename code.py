import cv2
import face_recognition
import os
import pickle
from tkinter import Tk, Label, Button, messagebox

# Set up data directory
data_dir = 'face_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def register_face(name):
    """Captures a face and stores it in the dataset with the given name"""
    video_capture = cv2.VideoCapture(0)
    registered = False

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if not registered:
                # Save face encoding for the user
                with open(f"{data_dir}/{name}.pkl", 'wb') as file:
                    pickle.dump(face_encoding, file)
                registered = True
                messagebox.showinfo("Success", f"Face registered for {name}")
        
        cv2.imshow("Registering Face", frame)

        if registered or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def authenticate_face():
    """Authenticates the user by comparing with stored face data"""
    known_face_encodings = []
    known_face_names = []

    # Load known faces
    for filename in os.listdir(data_dir):
        if filename.endswith(".pkl"):
            name = filename.split(".")[0]
            with open(f"{data_dir}/{filename}", 'rb') as file:
                known_face_encodings.append(pickle.load(file))
                known_face_names.append(name)

    video_capture = cv2.VideoCapture(0)
    authenticated = False

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                authenticated = True
                messagebox.showinfo("Authenticated", f"Welcome, {name}")
                break

        cv2.imshow("Authenticating", frame)

        if authenticated or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# GUI
def main_gui():
    root = Tk()
    root.title("Face Recognition Authentication")
    root.geometry("400x200")

    Label(root, text="Face Recognition Authentication", font=("Helvetica", 16)).pack(pady=10)

    Button(root, text="Register Face", command=lambda: register_face("User")).pack(pady=10)
    Button(root, text="Authenticate Face", command=authenticate_face).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
