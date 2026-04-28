import cv2
from ultralytics import YOLO
# import torch


def run_webcam_detection(
    model_path,
    camera_index=0,
    resize=None,
    person_only=False
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_human.mp4', fourcc, 20.0, (640, 480))
    if not cap.isOpened():
        raise ValueError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize:
            frame = cv2.resize(frame, resize)

        # Run detection
        if person_only:
            results = model(frame, classes=[0])  # class 0 = person
        else:
            results = model(frame)

        annotated = results[0].plot(conf=False, labels=False)
        cv2.imshow("YOLO Webcam Detection", annotated)
        out.write(annotated)  # Save frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # print("Torch version:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())
    # print("CUDA version:", torch.version.cuda)

    model = YOLO("yolov8n.pt")

    # model.train(
    #     data="../data/Person_detection_v1i_yolov8/data.yaml",
    #     epochs=15,
    #     imgsz=512,
    #     batch=2,
    #     workers=0,
    #     patience=5,
    #     name="yolo_human_detection",
    #     device=0
    # )
    run_webcam_detection(
        model_path="./best.pt",  # Change this to where <Your Model>.pt is located
        resize=(640, 480),
        person_only=True
    )
    # model = YOLO("../runs/detect/yolo_human_detection6/weights/best.pt")

    # img = cv2.imread("../data/human_test.jpg")
    # if img is None:
    #     raise ValueError("Image not found.")

    # results = model(img)
    # annotated = results[0].plot()

    # display = cv2.resize(annotated, (800, 600))
    # cv2.imshow("Detection", display)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
