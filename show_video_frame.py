import cv2

def show_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if ret:
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
    cap.release()

show_frame('./HybrIK/examples/child_running.mp4', 300)