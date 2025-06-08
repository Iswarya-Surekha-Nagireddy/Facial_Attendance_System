import cv2
import os
import face_recognition
import pickle
import pandas as pd

# Path to the images folder and CSV file
IMAGE_PATH = "C:/Users/nagir/Downloads/Face_Attendance_System/Images"
CSV_PATH = "C:/Users/nagir/Downloads/Face_Attendance_System/dataset/Face_Attendance_Dataset.csv"

# Load student details from CSV
students_data = pd.read_csv(CSV_PATH)

# Lists to store encodings and corresponding IDs
known_encodings = []
known_ids = []

# Loop through each folder (person's name_ID)
for folder_name in os.listdir(IMAGE_PATH):
    person_path = os.path.join(IMAGE_PATH, folder_name)
    
    if os.path.isdir(person_path):
        # Extract ID from the folder name
        folder_id = folder_name.split('_')[-1]
        
        # Validate ID from the CSV file
        if folder_id in students_data['StudentID'].astype(str).values:
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                # Load and encode the image
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                face_encodings = face_recognition.face_encodings(rgb_image)
                
                if face_encodings:  # Ensure face is detected
                    known_encodings.append(face_encodings[0])
                    known_ids.append(folder_id)

# Save encodings and IDs to a file for faster recognition
with open("trained_data.pkl", "wb") as file:
    pickle.dump({"encodings": known_encodings, "ids": known_ids}, file)

print("✅ Face training complete. Data saved as 'trained_data.pkl'")
