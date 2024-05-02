import winsound
from ultralytics import YOLO
import cv2
import pytesseract
import time
import validateID as et
import re
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# use the pre-trained dataset to track person
# model = YOLO('yolov8n.pt')
# model = YOLO(r'C:\Users\syshe\AI\assignment\StudentIDDetection\card_test20\weights\best.pt')
model = YOLO(r'.\card_test20\weights\best.pt')
# 11st training: epochs = 60
# 12nd training epochs = 100
# 13rd & 18th & 19th training epochs = 70, using back the test11 best.pt
# 14th & 16th training epochs = 80, using test13 best.pt
# 15th training, epochs = 70, using back the test13 bestp.pt
# 17th invalid run
# 20th training epochs = 70, using test18 best.pt

# training = model.train(data ='config.yaml', epochs = 70, name='card_test')
text = ""


def yolo_v8_detect(ocr_display, validated_display):
    global text
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale
    font_scale = 0.5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (700, 700), interpolation=cv2.INTER_LINEAR)

        model.predict(frame, conf=0.5)
        detections = model(frame)[0]
        detected = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, confident, class_id = detection
            if int(class_id) == 0 and confident > 0.80:

                # extract text from the student card
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                # convert roi to grayscale making teserract work btr
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray_roi)
                id_check, student_id = et.check_id(text)
                exp_date_check, exp_date = et.check_exp_date(text)
                if id_check and exp_date_check:
                    winsound.Beep(1000, 200)
                    validated_display.set(et.get_validated_info(student_id, exp_date))
                ocr_display.set(text)

                # box the card with its confidence level
                detected.append([x1, y1, x2, y2, confident])
                org = (int(x1), int(y1) - 5)  # the coordinate
                cv2.putText(frame, 'Confidence = {:.2f}'.format(confident), org, font,
                            font_scale, color, thickness, cv2.LINE_AA)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        #             time.sleep(1)

        cv2.imshow('Phone camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)

    # Run batched inference on a list of images
    # import os

    # # Define the path to the directory containing the images
    # path = "C:\\Users\\syshe\\AI\\assignment\\test"

    # # List of image filenames
    # image_files = ['51.jpg', '80.jpg', '70.jpg', '83.jpg']

    # # Generate full paths for each image
    # image_paths = [os.path.join(path, img_file) for img_file in image_files]

    # # Run batched inference on the list of images
    # results = model(image_paths)  # This will return a list of Results objects

    # # Process results list
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     result.show()  # display to screen
# test_string = "SIA YEONG SHENG == 23WMR09471 -----=ad EXPIRY DATE: 31-08-2025 200010123"
