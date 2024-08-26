import cv2
import numpy as np
import torch
import json
import math
import time

from boxmot import DeepOCSORT
from pathlib import Path
from absl import app, flags
from absl.flags import FLAGS
from shapely.geometry import Point, Polygon
from collections import defaultdict, deque

# from models.common import DetectMultiBackend, AutoShape

# Detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load(
    "yolov5",
    "custom",
    path="yolov5/weights/best.pt",
    source="local"  # Local repo
)
model.conf = 0.25
model.max_det = 1000
model.eval()
# model.cuda()

# Tracking model
tracker = DeepOCSORT(
    model_weights=Path('yolov5/osnet_x0_25_msmt17.pt'),  # Which ReID model to use
    device='cpu',
    fp16=False,
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
                                 reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
    

def draw_box(img, tracks, distance_cal, num_frame, area_cal):
    global colors
    global class_names
    global roi_polygon
    global track_object
    global speed_function
    global fps

    num_car, num_bike, num_bus, num_truck = 0, 0, 0, 0
    avg_speed = 0

    xyxys = tracks[:, 0:4].astype(int)
    ids = tracks[:, 4].astype(int)
    conf = tracks[:, 5]
    clss = tracks[:, 6].astype(int)
    inds = tracks[:, 7].astype(int)  # float64 to int

    for idx in range(tracks.shape[0]):
        left, top, right, bottom = xyxys[idx]
        x_center, y_center = (left + right) / 2, (top + bottom) / 2
        point = Point(x_center, y_center)
        is_inside = roi_polygon.contains(point)

        if is_inside:
            if track_object.get(ids[idx]) is not None:
                to = track_object[ids[idx]]
                object_speed = -1

                his_fram = to[1]
                pivot = top
                new_distance = to[0]
                distance = distance_cal(pivot)

                if to[0] != -1:
                    delta_distance = abs(distance - new_distance)
                    if delta_distance >= 4.0 or num_frame - his_fram > 2 * fps:
                        object_speed = speed_function(delta_distance, to[1], num_frame)
                        his_fram = num_frame
                        new_distance = distance
                    else:
                        object_speed = to[2]
                else:
                    new_distance = distance
                    track_object[ids[idx]] = [new_distance, his_fram, object_speed]

                if object_speed != -1:
                    avg_speed += object_speed / len(track_object)
                    cv2.putText(
                        img,
                        "%.2f" % object_speed,
                        (int(x_center) - 12, int(y_center)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colors[clss[idx]],
                        1,
                    )

            else:
                track_object[ids[idx]] = [-1, num_frame, -1]

            text_clss = clss[idx]
            if text_clss == 0:
                num_car += 1
            elif text_clss == 1:
                num_bike += 1
            elif text_clss == 2:
                num_bus += 1
            elif text_clss == 3:
                num_truck += 1

            label = "{}".format(ids[idx])
            # cv2.rectangle(img, (left, top), (right, bottom), colors[text_clss], 1)
            # cv2.putText(img, label, (left + 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[text_clss], 2)

    # occupancies = area_cal(num_car, num_bike, num_bus, num_truck)
    # if avg_speed < 5 and occupancies > 0.8:
    #     cv2.putText(img, "TAC DUONG", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    # else:
    #     cv2.putText(img, "KHONG TAC", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # cv2.putText(img, "num_car:{}".format(num_car), (630, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[0], 2)
    # cv2.putText(img, "num_bike:{}".format(num_bike), (630, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[1], 2)
    


def get_pixel(rw, ppa, aw, delta, rh, vh, vw, diameter,  isLeft = False):
    # rw = 1.1*rw
    alpha = (180 / math.pi) * math.acos(rh * math.tan(aw) / rw)
    if alpha < delta:
        dis = rh * math.tan(aw) / math.cos(delta * math.pi / 180)
        percent_ship = (dis - rw) / dis
        x = vw - percent_ship * vw / 2
        if isLeft:
            x = percent_ship * vw / 2
        point = (x / vw, 1)
    else:
        px = (alpha - delta) * ppa
        x = vw
        if isLeft:
            x = 0
        point = (x / vw, (vh - px) / vh)

    top_alpha = math.atan(diameter / rh) * 180 / math.pi
    top_px = int((top_alpha - delta) * ppa)
    if top_px > vh:
        top_alpha = delta + vh / ppa
        top_px = vh
    distance = rh * math.tan(aw) / math.cos(top_alpha * math.pi / 180)
    ratio = (distance - rw) / distance
    x = vw - ratio * vw / 2
    if isLeft:
        x = ratio * vw / 2
    return (point, (x / vw, (vh - top_px) / vh))


def get_roi_points(
    touch_angle, 
    h_angle, 
    real_height, 
    left_real_width, 
    right_real_width, 
    width, 
    height, 
    diameter,
    ppa
):
    width_angle = h_angle * math.pi / 180
    point_left = get_pixel(left_real_width, ppa, width_angle, touch_angle, 
                           real_height, height, width, diameter, True)
    point_right = get_pixel(right_real_width, ppa, width_angle, touch_angle, 
                            real_height, height, width, diameter,
                            False)
    lst = [point_left[0], point_left[1], point_right[1]]
    t1 = point_left[0][1]
    t2 = point_right[0][1]
    if t1 == t2:
        lst.append(point_right[0])
        lst.append(point_left[0])
    else:
        if t2 > t1:
            rtl = (point_right[0], (point_left[0][0], t2))
        elif t1 > t2:
            rtl = (point_right[0], (point_right[0][0], t1))
            min = t2
        lst.append(rtl[0])
        lst.append(rtl[1])
        lst.append(point_left[0])
    bottom = touch_angle + (height - t1 * height) / ppa
    top = touch_angle + (height - point_left[1][1] * height) / ppa
    diameter = (math.tan((math.pi / 180) * top) * real_height - 
                math.tan((math.pi / 180) * bottom) * real_height)
    return lst, diameter


def search_pivot(points):
    top, bottom, right, left = int(1e6), 0, 0, int(1e6)
    for points in points:
        if points[1] < top:
            top = points[1]
        elif points[1] > bottom:
            bottom = points[1]
        if points[0] < left:
            left = points[0]
        elif points[0] > right:
            right = points[0]
    return top, bottom, right, left


def region_of_interested(raw_rois, width, height):
    roi_points = []
    for point in raw_rois:
        roi_points.append((int(point[0] * width), int(point[1] * height)))
    return roi_points, np.asarray(raw_rois)


def speed_estimation(fps=30):
    return lambda x, y, y_: 3.6 * fps * abs(x) / (y_ - y)


def occupancy_estimation(car, bike, bus, truck, area):
    return lambda x, y, z, h: (x * car + y * bike + z * bus + h * truck) / area


# Perspective Transformation
def draw_box_2(
        img,
        tracks,
        distance_cal,
        num_frame,
        area_cal,
        source,
        target
):
    global colors
    global class_names
    global roi_polygon
    global track_object
    global speed_function
    global fps
    global coordinates
    global view_transformer

    xyxys = tracks[:, 0:4].astype(int)
    ids = tracks[:, 4].astype(int)
    conf = tracks[:, 5]
    clss = tracks[:, 6].astype(int)
    inds = tracks[:, 7].astype(int)  # float64 to int

    num_car, num_bike, num_bus, num_truck = 0, 0, 0, 0
    avg_speed = 0

    for idx in range(tracks.shape[0]):
        left, top, right, bottom = xyxys[idx]
        x_center, y_center = (left + right) / 2, (top + bottom) / 2
        center_point = Point(x_center, y_center)
        is_inside = roi_polygon.contains(center_point) # check if in roi_polygon

        if is_inside:
            coord_object = [(x_center, y_center)]
            coord_object = np.array(coord_object)
            points = view_transformer.transform_points(points=coord_object).astype(int)

            # Store the transformed coordinates
            coordinates[ids[idx]].append((points[0][0], points[0][1], num_frame, -1))

            # Wait to have enough data
            if len(coordinates[ids[idx]]) >= (4 * fps):
                # Calculate the speed
                coordinate_start = coordinates[ids[idx]][-1]
                coordinate_end = coordinates[ids[idx]][2 * fps]
                distance = ((coordinate_start[0] - coordinate_end[0])**2 +
                            (coordinate_start[1] - coordinate_end[1])**2)**0.5
                eta = abs(coordinate_start[2] - coordinate_end[2])
                if distance >= 0.5:
                    time = float(eta / fps)
                    speed = (distance / time) * 3.6
                    vex = coordinates[ids[idx]]
                    coordinates[ids[idx]][-1] = (points[0][0], points[0][1], num_frame, speed)
                    avg_speed += (speed / len(tracks))

                vex = coordinates[ids[idx]][-1]
                check_speed = vex[3]
                if int(check_speed) != -1:
                    cv2.putText(img, "{:.2f}".format(check_speed), (left + 2, top - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            text_clss = clss[idx]
            if text_clss == 0:
                num_car += 1
            elif text_clss == 1:
                num_bike += 1
            elif text_clss == 2:
                num_bus += 1
            elif text_clss == 3:
                num_truck += 1

            cv2.rectangle(img, (left, top), (right, bottom), colors[clss[idx]], 2)

    occupancies = area_cal(num_car, num_bike, num_bus, num_truck)
    if avg_speed < 5 and occupancies > 0.9:
        cv2.putText(img, 'fps:{}'.format(fps), (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, 'tac duong', (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(img, 'fps:{}'.format(fps), (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, 'khong tac', (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.putText(img, "num_car:{}".format(num_car), (630, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[0], 2)
    cv2.putText(img, "num_bike:{}".format(num_bike), (630, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[1], 2)


def read_ccfg(height, width, diameter):
    # file_path = FLAGS.ccfg
    file_path = './cfg/cfg.txt'

    # print('use camera config file: ' + file_path)
    with open(file_path, 'r') as f:
      data = [float(i.strip()) for i in f.readlines()]

    # real_height = data[0]
    # left_width = data[1]
    # right_width = data[2]
    # base_angle = data[3]
    # delta_angle = data[4]
    # h_angle = data[5]
    real_height = cv2.getTrackbarPos('real_height', 'image') 
    left_width = cv2.getTrackbarPos('left_width', 'image')
    right_width = cv2.getTrackbarPos('right_width', 'image')
    base_angle = cv2.getTrackbarPos('base_angle', 'image') 
    delta_angle = cv2.getTrackbarPos('delta_angle', 'image')
    h_angle = cv2.getTrackbarPos('h_angle', 'image')
    top_rank = data[6]
    ppa = 0.5*height/delta_angle
    app = 2*delta_angle/height
    touch_angle = abs(base_angle-delta_angle)
    roi_points, diameter = get_roi_points(touch_angle, h_angle, real_height, 
                                          left_width, right_width, width, height, 
                                          diameter, ppa)
    left_area = diameter*left_width
    right_area = diameter*right_width

    calc_func = lambda x: (
        math.tan(
            (touch_angle + app * (height - x)) * math.pi / 180
        ) * real_height
    )

    return calc_func, roi_points, left_area, right_area


def main():
    video_path = '/home/minhthanh/directory_env/object_tracking2/auto-traffic-congestion/input_video/QT4.mp4'
    # video_path = '/home/minhthanh/Downloads/video1_2.mp4'
    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    global fps
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    conf_threshold = 0.25
    tracking_class = [0, 1, 2, 3]  # 0:car, 1:motorbike, 2:bus, 3:truck
    
    global colors, class_names
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    
    with open("yolov5/data_txt/classes.names") as f:
        class_names = f.read().strip().split('\n')

    # diameter = 30
    # distance_cal, roi_points, left_area, right_area = read_ccfg(height, width, diameter)
    # roi_points, raw_rois = region_of_interested(roi_points, width, height)
    # top, bottom, right, left = search_pivot(roi_points)
    
    global roi_polygon
    # roi_polygon = Polygon(roi_points)

    # Save object information: (distance, frame, speed)
    global track_object
    track_object = dict()

    global speed_function
    speed_function = speed_estimation(fps=fps)
    
    num_frame = 0

    # Save object coordinates for 4 seconds (x, y, frame)
    global coordinates
    coordinates = defaultdict(lambda: deque(maxlen=4 * fps))

    # true_area = left_area + right_area
    # area_cal = occupancy_estimation(13, 3, 40, 60, true_area)

    # SOURCE = np.array([roi_points[i] for i in range(4)])

    # # Setup manual --> can be automated
    # TARGET = np.array([[0, 19],
    #                    [0, 0],
    #                    [11, 0],
    #                    [11, 19]])

    # global view_transformer
    # view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # output_file = 'output_video.mp4'

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    # out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

  
    def nothing(x):
        pass 

    cv2.namedWindow('image') 
    cv2.createTrackbar('real_height', 'image', 0, 20, nothing) 
    cv2.createTrackbar('left_width', 'image', 0, 20, nothing)
    cv2.createTrackbar('right_width', 'image', 0, 20, nothing)
    cv2.createTrackbar('base_angle', 'image', 0, 180, nothing)
    cv2.createTrackbar('delta_angle', 'image', 0, 180, nothing)
    cv2.createTrackbar('h_angle', 'image', 0, 180, nothing)

    while True:
        
        # get current positions of all Three trackbars 
        start = time.time()
        ret, frame = vid.read()
        if not ret:
            break

        try:
            diameter = 30
            distance_cal, roi_points, left_area, right_area = read_ccfg(height, width, diameter)
            roi_points, raw_rois = region_of_interested(roi_points, width, height)
            top, bottom, right, left = search_pivot(roi_points)

            roi_polygon = Polygon(roi_points)
            true_area = left_area + right_area
            area_cal = occupancy_estimation(13, 3, 40, 60, true_area)


            real_height = cv2.getTrackbarPos('real_height', 'image') 
            left_width = cv2.getTrackbarPos('left_width', 'image')
            right_width = cv2.getTrackbarPos('right_width', 'image')
            base_angle = cv2.getTrackbarPos('base_angle', 'image') 
            delta_angle = cv2.getTrackbarPos('delta_angle', 'image')
            h_angle = cv2.getTrackbarPos('h_angle', 'image')


            pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            file_path = './cfg/cfg.txt'

            # print('use camera config file: ' + file_path)
            with open(file_path, 'r') as f:
                data = [float(i.strip()) for i in f.readlines()]

            # real_height = data[0]
            # left_width = data[1]
            # right_width = data[2] 
            # base_angle = data[3]
            # delta_angle = data[4]
            # h_angle = data[5]

            cv2.putText(frame, 'real_height:{}'.format(real_height), (450, 40), cv2.FONT_HERSHEY_DUPLEX, 1, 0, 1)
            cv2.putText(frame, 'left_width:{}'.format(left_width), (450, 65), cv2.FONT_HERSHEY_DUPLEX, 1, 0, 1)
            cv2.putText(frame, "right_width:{}".format(right_width), (450, 90), cv2.FONT_HERSHEY_DUPLEX, 1, 0, 1)
            cv2.putText(frame, "base_angle:{}".format(base_angle), (450, 115), cv2.FONT_HERSHEY_DUPLEX, 1, 0, 1)
            cv2.putText(frame, "delta_angle:{}".format(delta_angle), (450, 140), cv2.FONT_HERSHEY_DUPLEX, 1, 0, 1)
            cv2.putText(frame, "h_angle:{}".format(h_angle), (450, 165), cv2.FONT_HERSHEY_DUPLEX, 1, 0, 1)

            # Inference detection
            # frame = cv2.resize(frame, (1024, 1024))
            results = model(frame)
            data = results.pandas().xyxy[0].to_json(orient="records")
            json_data = json.loads(data)

            if not json_data:
                continue

            # Output has to be N x (x, y, x, y, conf, cls)
            num_frame += 1
            detect = []
            for record in json_data:
                confidence = round(record['confidence'], 2)
                class_id = record['class']
                left = int(record['xmin'])
                top = int(record['ymin'])
                right = int(record['xmax'])
                bottom = int(record['ymax'])
                # cv2.rectangle(frame, (left, top), (right, bottom), colors[class_id], 2)
                if class_id in tracking_class and confidence >= conf_threshold:
                    detect.append([left, top, right, bottom, confidence, class_id])

            detect = np.array(detect)
            if len(detect) > 0:
                tracks = tracker.update(detect, frame)  # M x (x, y, x, y, id, conf, cls, ind)
            else:
                detect = np.empty((0, 6))  # Empty N x (x, y, x, y, conf, cls)
                tracks = tracker.update(detect, frame)  # M x (x, y, x, y, id, conf, cls, ind)

            if tracks is not None and len(tracks) > 0:
                draw_box(frame, tracks, distance_cal, num_frame, area_cal)
                # draw_box_2(frame, tracks, distance_cal, num_frame, area_cal, SOURCE, TARGET)
        except:
            pass

        # Break on pressing q or space
        # out.write(frame)
        cv2.imshow('BoxMOT detection', frame)
        print("time per frame: {}s".format(time.time() - start))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break

    vid.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
