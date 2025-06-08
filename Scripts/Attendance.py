import cv2
import face_recognition
import pickle
import pandas as pd
import requests
from datetime import datetime

# Load the trained data
with open("trained_data.pkl", "rb") as file:
    data = pickle.load(file)
    known_encodings = data["encodings"]
    known_ids = data["ids"]

# Load student details from CSV
students_data = pd.read_csv("C:/Users/nagir/Downloads/Face_Attendance_System/dataset/Face_Attendance_Dataset.csv")

# Google Sheets API URL
URL = "https://script.google.com/macros/s/AKfycbxw20eTUc894pAA3zpa-ky1Zq7gXCZ_lmS-dKWer6pdlOH6gdb1wrW0usbNdO_OT5c/exec"

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Track attendance for the current day
marked_today = set()
current_date = datetime.now().strftime("%d/%m/%Y")

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Check if the date has changed (resets marked list)
    today_date = datetime.now().strftime("%d/%m/%Y")
    if today_date != current_date:
        marked_today.clear()
        current_date = today_date

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            student_id = known_ids[best_match_index]
            student_info = students_data[students_data['StudentID'] == student_id].iloc[0]
            name = student_info['Name']

            # Check if already marked today
            if student_id not in marked_today:
                now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                # Send data to Google Sheets
                params = {"name": name, "id": student_id, "status": "Present", "datetime": now}
                response = requests.get(URL, params=params)

                if "Success" in response.text:
                    print(f"✅ Attendance marked for: {name}")
                    marked_today.add(student_id)  # Add to marked list
                else:
                    print(f"⚠️ {response.text}")
            else:
                print(f"🔄 {name} is already marked present today.")

        else:
            name = "Unknown"

        # Draw rectangle and label on the detected face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition Attendance System", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
