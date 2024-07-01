import numpy as np
import cv2
import tensorflow as tf
from utils.configurations import KEYPOINT_DICT as kp_names
from utils.predictor import preprocess, get_prediction

videofile = input('Enter video path: ')
is_child = input('Is the person in the video a child? (y/n): ').lower() == 'y'
trial_speed = input('Is the trial slow or fast? (s/f): ').lower()

def build_interpreter(path):
    model = tf.keras.models.load_model(path)
    return model.signatures["serving_default"]

interpreter = build_interpreter(path='model')
cap = cv2.VideoCapture(videofile)

prev_left_ankle_y = None
prev_right_ankle_y = None
if is_child == 'y':
    threshold = 15
else:
    threshold = 10
score = 0

time_intervals = {
    ('f', True): (5, 9), 
    ('f', False): (3, 24),
    ('s', True): (5, 12),
    ('s', False): (3, 23)
}

try:
    start_time, end_time = time_intervals[(trial_speed, is_child)]
except KeyError:
    print("Invalid input for trial speed. Please enter 's' for slow or 'f' for fast.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor, frame, _ = preprocess(frame[...,::-1], input_size=(512, 512))
    fw, fh, _ = frame.shape
    kpts, _, _, _, _ = get_prediction(input_tensor, interpreter, from_class=1)
    kpts = kpts[...,::-1] * np.array([fw, fh])

    left_ankle = kpts[0, kp_names['left_ankle']]
    right_ankle = kpts[0, kp_names['right_ankle']]
    left_ankle_y = left_ankle[1]
    right_ankle_y = right_ankle[1]

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    if start_time <= current_time <= end_time:
        if prev_left_ankle_y is not None:
            if (prev_left_ankle_y - left_ankle_y > threshold) and (left_ankle_y - prev_left_ankle_y < -threshold):
                score += 1

        if prev_right_ankle_y is not None:
            if (prev_right_ankle_y - right_ankle_y > threshold) and (right_ankle_y - prev_right_ankle_y < -threshold):
                score += 1

    prev_left_ankle_y = left_ankle_y
    prev_right_ankle_y = right_ankle_y

    for kp in kpts[0]:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Final score:", score)