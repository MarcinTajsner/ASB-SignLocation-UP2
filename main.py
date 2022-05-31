import cv2
from attrs import define
from argparse import ArgumentParser
from detection_info import get_camera_info, get_classes
from inference_engine import InferenceEngine
from time import time, sleep


RESOLUTION_X = 1920
RESOLUTION_Y = 1080
FPS = 30
FOCAL = get_camera_info()["focal_length"]
CLASSES = get_classes()
IE = InferenceEngine(
    model_path="models/openvino/yolov5n6_1280_32_200/yolov5n6_1280_32_200.xml",
    weights_path="models/openvino/yolov5n6_1280_32_200/yolov5n6_1280_32_200.bin",
    device="CPU",
    confidence_threshold=0.5
)


@define
class ImageProccesing:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    obj_info: dict
    distance: int = None

    def procces_frame(self, frame):
        cv2.rectangle(frame,
                      (self.xmin, self.ymin),
                      (self.xmax, self.ymax),
                      self.obj_info['color'],
                      2)
        cv2.putText(frame, f"{self.obj_info['name']}", (self.xmax, self.ymin),
                    cv2.FONT_HERSHEY_PLAIN, 2, self.obj_info['color'], 2, cv2.LINE_AA)
        if self.distance:
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


def single_frame_procces(frame):
    cords = IE.detect_from_frame(frame)
    objects_to_add_to_frame = []
    for cord in cords:
        class_info = CLASSES[cord["class"]]
        if class_info["size"] is not None:
            objects_to_add_to_frame.append(
                ImageProccesing(int(cord["xmin"]), int(cord["ymin"]), int(cord["xmax"]), int(cord["ymax"]), class_info, calculate_distance(cord)))
        else:
            objects_to_add_to_frame.append(ImageProccesing(int(cord["xmin"]), int(cord["ymin"]), int(
                cord["xmax"]), int(cord["ymax"]), class_info))
    return objects_to_add_to_frame


def start(use_camera: bool = False, frame_rate: int = 1, viedo_path: str = 'test_video.mp4', fps: int = 30, save_viedo: bool = False, live_display: bool = True):
    if fps < 0 or fps > 60:
        raise Exception("Fps could be only in range <0;60>")
    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(viedo_path)
    RESOLUTION_X, RESOLUTION_Y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if save_viedo:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps,
                              (RESOLUTION_X, RESOLUTION_Y))
    single_frame_time = 1 / fps
    curr_frame = frame_rate - 1
    objects_to_add_to_frame = []
    frame_number, skipped_frames = 0, 0
    FPS = 30
    while cap.isOpened():
        start = time()
        if frame_number >= fps:
            FPS = fps - skipped_frames
            frame_number, skipped_frames = 0, 0
        curr_frame += 1
        frame_number += 1
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if curr_frame >= frame_rate:
            objects_to_add_to_frame = single_frame_procces(frame)
            curr_frame = 0
        for obj in objects_to_add_to_frame:
            obj.procces_frame(frame)
        cv2.putText(frame, f"{FPS} FPS", (RESOLUTION_X - 60, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv2.LINE_AA)
        if save_viedo:
            out.write(frame)

        frame_time = time() - start
        frame_diff = single_frame_time - frame_time
        if frame_diff > 0:
            sleep(frame_diff)
        elif not save_viedo:
            # Skip frames
            to_skip = round(frame_time / single_frame_time)
            skipped_frames += to_skip
            frame_number += to_skip
            sleep(to_skip // 2 * single_frame_time)
            for _ in range(to_skip):
                ret, _ = cap.read()

        if live_display:
            cv2.imshow("fr", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    if save_viedo:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


parser = ArgumentParser(description="Sign recognition.")
parser.add_argument("--use_cam", action='store_true',
                    help="specify that video should be captured from camera.")
parser.add_argument("-frame_rate", type=int, default=1,
                    help="specify every how many frames the recognition is to take place.")
parser.add_argument("-video_path", default="", metavar="PATH",
                    help="specify path to viedo that should be used for recognition.")
parser.add_argument("--live_display", action='store_true',
                    help="if true video will be displayed live.")
parser.add_argument("--save_video", action='store_true',
                    help="if true rendred video will be saved.")
parser.add_argument("-fps", default=30, type=int,
                    help="FPS of display/output video.")
args = vars(parser.parse_args(["-frame_rate", "3"]))
args = parser.parse_args()
# args = parser.parse_args()
# start(args.use_cam, args.frame_rate, args.video_path,
#       args.fps, args.save_video, args.live_display)
start()
