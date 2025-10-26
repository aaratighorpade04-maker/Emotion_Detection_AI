import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Fix for list/dict issue
        if isinstance(result, list):
            emotion = result[0]['dominant_emotion']
        else:
            emotion = result['dominant_emotion']

        # Display on frame
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()