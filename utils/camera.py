import cv2


class CameraSource:
    def __init__(self, device_index=0, width=640, height=480, api=cv2.CAP_DSHOW):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.api = api
        self.cap = None
        self.last_frame = None
        self.last_error = None

    def open(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_index, self.api)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            else:
                self.last_error = f"Unable to open camera {self.device_index}"
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.open():
            return False, None
        success, frame = self.cap.read()
        if not success or frame is None:
            self.last_error = "Frame read failed"
            return False, None
        self.last_frame = frame
        return True, frame

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    @staticmethod
    def to_rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def annotate_status(frame, label, color=(255, 255, 255)):
        annotated = frame.copy()
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return annotated
