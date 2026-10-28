import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import os

# Load known faces
def load_known_faces():
    known_encodings = []
    known_names = []
    dataset_path = "dataset"
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person)
    
    return known_encodings, known_names

# Mark attendance
def mark_attendance(name):
    log_dir = "attendance_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{log_dir}/attendance_{today}.csv"
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Check if already marked today
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == name:
                    return  # Already present
    
    # Mark attendance
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, current_time])
    print(f"Attendance marked for {name} at {current_time}")

# Main function
def main():
    known_encodings, known_names = load_known_faces()
    print(f"Loaded {len(known_names)} known faces.")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            name = "Unknown"
            if len(matches) > 0 and len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name)
            
            # Draw rectangle and label
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("Face Attendance System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()