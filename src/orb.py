import cv2

# 1. Initialize input
cap = cv2.VideoCapture("./sussy.MP4")
orb = cv2.ORB_create(nfeatures=1000)

# Define the frame size once to keep things consistent
width, height = 960, 540

# 2. Update VideoWriter to match your resize dimensions
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('orbt.mp4', fourcc, 20.0, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Resize to the EXACT same dimensions as the VideoWriter
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. ORB logic
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 5. Draw
    frame = cv2.drawKeypoints(frame, keypoints, None,
                              color=(0, 255, 0), flags=0)

    # 6. Display and Save
    cv2.imshow('Real-time ORB', frame)
    out.write(frame)  # This will work now because sizes match!

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Crucial: Don't forget to release 'out' to finish the file!
cv2.destroyAllWindows()
