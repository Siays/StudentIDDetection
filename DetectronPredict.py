from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import pytesseract
import cv2
import re
import validateID as et
import winsound

# set the path for tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detectronDetect(ocr_display):
    # Load config from a config file
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = './detectronOutput/colab_896v2.1.pth'
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # Create predictor instance
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(1)

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_LINEAR)

        # Perform prediction
        outputs = predictor(frame)

        threshold = 0.5

        # Display predictions
        preds = outputs["instances"].pred_classes.tolist()
        scores = outputs["instances"].scores.tolist()
        bboxes = outputs["instances"].pred_boxes

        for j, bbox in enumerate(bboxes):
            bbox = bbox.tolist()

            score = scores[j]
            pred = preds[j]

            if score > threshold:
                x1, y1, x2, y2 = [int(i) for i in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

                # Extract the text from the detected object
                object_image = frame[y1:y2, x1:x2]
                gray_object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray_object_image)
                if et.check_id(text) and et.check_exp_date(text):
                    winsound.Beep(1000, 200)
                ocr_display.set(text)
                # cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('image', frame)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()



