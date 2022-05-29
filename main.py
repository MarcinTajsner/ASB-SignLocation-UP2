import cv2
from attrs import define
from argparse import ArgumentParser
from detection_info import get_camera_info, get_classes
from inference_engine import InferenceEngine

RESOLUTION_X = 1920
RESOLUTION_Y = 1080
FOCAL = get_camera_info()["focal_length"]
CLASSES = get_classes()
IE = InferenceEngine(
    model_path = "models/openvino/yolov5n6_1280_32_200/yolov5n6_1280_32_200.xml",
    weights_path = "models/openvino/yolov5n6_1280_32_200/yolov5n6_1280_32_200.bin",
    device = "HDDL",
    confidence_threshold = 0.5
)

@define
class ImageProccesing:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    distance: int
    obj_info: dict

    def procces_frame(self, frame):
        cv2.rectangle(frame,
                      (self.xmin, self.ymin),
                      (self.xmax, self.ymax),
                      self.obj_info['color'],
                      2)
        cv2.putText(frame, f"{self.obj_info['name']}", (self.xmax, self.ymin),
                    cv2.FONT_HERSHEY_PLAIN, 2, self.obj_info['color'], 2, cv2.LINE_AA)
        cv2.putText(frame, f"{self.distance:.2f} m", (self.xmax, self.ymax - (abs(self.ymax - self.ymin)//2)),
                    cv2.FONT_HERSHEY_PLAIN, 2, self.obj_info['color'], 2, cv2.LINE_AA)


def calculate_distance(cord: dict):
    real_obj_size = CLASSES[cord["class"]]["size"]
    if abs(cord["ymin"] - cord["ymax"]) > abs(cord["xmin"] - cord["xmax"]):
        obj_on_img_size = abs(cord["xmin"] - cord["xmax"])
        sensor_size = get_camera_info()["sensor_size_x"]
        resolution = RESOLUTION_X
    else:
        obj_on_img_size = abs(cord["ymin"] - cord["ymax"])
        sensor_size = get_camera_info()["sensor_size_y"]
        resolution = RESOLUTION_Y
    return (FOCAL * real_obj_size * resolution) / \
        (obj_on_img_size * sensor_size) / 1000


def start(use_camera: bool = False, frame_rate: int = 2, viedo_path: str = 'test_video.mp4'):
    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(viedo_path)
    curr_frame = frame_rate - 1
    objects_to_add_to_frame = []
    while cap.isOpened():
        curr_frame += 1
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if curr_frame == frame_rate:
            cords = IE.detect_from_frame(frame)
            objects_to_add_to_frame = []
            for cord in cords:
                class_info = CLASSES[cord["class"]]
                if class_info["size"] is not None:
                    objects_to_add_to_frame.append(
                        ImageProccesing(int(cord["xmin"]), int(cord["ymin"]), int(cord["xmax"]), int(cord["ymax"]), calculate_distance(cord), class_info))
            curr_frame = 0
        for obj in objects_to_add_to_frame:
            obj.procces_frame(frame)
        cv2.imshow("fr", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


parser = ArgumentParser(description="Sign recognition.")
parser.add_argument("--use_cam", action='store_true',
                    help="specify that video should be captured from camera.")
parser.add_argument("-frame_rate", type=int, default=1,
                    help="specify every how many frames the recognition is to take place.")
parser.add_argument("-video_path", default="", metavar="PATH",
                    help="specify path to viedo that should be used for recognition.")
args = vars(parser.parse_args(["-frame_rate", "3"]))
args = parser.parse_args()
#start(args.use_cam, args.frame_rate, args.video_path)
start()