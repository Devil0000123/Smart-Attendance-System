import cv2
import os
from datetime import datetime

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def enroll_student(student_name, num_images=30):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    student_dir = os.path.join(DATASET_DIR, student_name)
    os.makedirs(student_dir, exist_ok=True)

    print(f"Enrolling {student_name}... Press 'q' to stop early.")
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Show live feed
        cv2.imshow("Enroll - Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            img_path = os.path.join(student_dir, f"{student_name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
            print(f"Saved {count}/{num_images} images")

    cap.release()
    cv2.destroyAllWindows()
    print("Enrollment done.")

if __name__ == "__main__":
    name = input("Enter student name (e.g., 'Prince'): ").strip()
    if name:
        enroll_student(name, num_images=30)
    else:
        print("No name provided.")