import cv2

net = cv2.dnn.readNet(
    r"C:\Users\11520\PycharmProjects\pytorch\object_detection_crash_course\dnn_model\yolov4-tiny.cfg",
    r"C:\Users\11520\PycharmProjects\pytorch\object_detection_crash_course\dnn_model\yolov4-tiny.weights")
cap = cv2.VideoCapture(0)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open(r"C:\Users\11520\PycharmProjects\pytorch\object_detection_crash_course\dnn_model\classes.txt",
          "r") as flie_object:
    for class_name in flie_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print(classes)

while True:
    ret, frame = cap.read()
    (class_ids, score, bboxes) = model.detect(frame)
    for class_ids, score, bbox in zip(class_ids, score, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_ids]
        print(x, y, w, h)
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
    print("clss_ids", class_ids)
    print("scores", score)
    print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
cv2.namedWindow("Frame", 10)

cv2.resizeWindow("Frame", 900, 700)