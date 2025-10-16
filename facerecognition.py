import cv2
import face_recognition
import numpy as np
import os

# --- Constants ---
ENCODINGS_FILE = "known_faces.npy"
NAMES_FILE = "known_names.txt"
RECOGNITION_TOLERANCE = 0.5  # stricter to avoid false positives

# --- Helper Functions ---
def load_known_faces(encodings_path, names_path):
    """Loads face encodings and names from files."""
    known_face_encodings = []
    known_face_names = []
    if os.path.exists(encodings_path) and os.path.exists(names_path):
        known_face_encodings = list(np.load(encodings_path, allow_pickle=True))
        with open(names_path, "r") as f:
            known_face_names = [line.strip().lower() for line in f.readlines()]
        print(f"Loaded {len(known_face_names)} known faces.")
    return known_face_encodings, known_face_names

def save_known_faces(encodings, names, encodings_path, names_path):
    """Saves face encodings and names to files."""
    np.save(encodings_path, np.array(encodings, dtype=object))
    with open(names_path, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"Saved {len(names)} known faces.")

# --- Main Function ---
def main():
    known_face_encodings, known_face_names = load_known_faces(ENCODINGS_FILE, NAMES_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    print("Starting webcam feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]

                    if best_distance < RECOGNITION_TOLERANCE:
                        name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Draw results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.9, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            if face_names.count("Unknown") == 1 and len(face_encodings) == 1:
                new_face_encoding = face_encodings[0]
                name = input("Enter the name for the new face and press Enter: ").strip().lower()

                if name:
                    # Allow multiple encodings for the same person
                    known_face_encodings.append(new_face_encoding)
                    known_face_names.append(name)
                    save_known_faces(known_face_encodings, known_face_names, ENCODINGS_FILE, NAMES_FILE)
                else:
                    print("Invalid name. Face not saved.")
            elif len(face_encodings) == 0:
                print("No face detected. Cannot save.")
            else:
                print("Multiple faces detected. Ensure only one 'Unknown' face is visible.")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == '__main__':
    main()
