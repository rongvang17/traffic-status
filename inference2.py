import cv2
import numpy as np
import torch
import json
import math

from boxmot import DeepOCSORT
from pathlib import Path
from absl import app, flags
from absl.flags import FLAGS
from shapely.geometry import Point, Polygon
# from models.common import DetectMultiBackend, AutoShape

flags.DEFINE_string('ccfg', './cfg/cfg2.txt', 'path to camera config file')

# detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load("yolov5", "custom", path="/home/minhthanh/directory_env/object_tracking/yolov5/weights/best.pt", source="local")  # local repo
model.conf = 0.4
model.max_det = 1000
# model.cuda()

# tracking model
tracker = DeepOCSORT(
    model_weights=Path('yolov5/osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cpu',
    fp16=False,
)


def draw_box(img, tracks, distance_cal, num_frame):
    global colors
    global class_names
    global roi_polygon
    global track_object
    global speed_function
    global fps
    num_car = 0
    num_bike = 0
    num_bus = 0
    num_truck = 0
    xyxys = tracks[:, 0:4].astype('int')
    ids = tracks[:, 4].astype('int')
    conf = tracks[:, 5]
    clss = tracks[:, 6].astype('int')
    inds = tracks[:, 7].astype('int') # float64 to int

    for idx in range(tracks.shape[0]):
        left , top, right, bottom = xyxys[idx]
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
                    delta_distance = abs(distance-new_distance)
                    if delta_distance >= 4.0 or num_frame-his_fram > 2*fps:
                        object_speed = speed_function(delta_distance, to[1], num_frame)
                        his_fram = num_frame
                        new_distance = distance
                    else:
                        object_speed = to[2]
                else:
                    new_distance = distance
                    track_object[ids[idx]] = [new_distance, his_fram, object_speed]

                if object_speed != -1:
                    cv2.putText(img, "%.2f" % (object_speed), (left + 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[clss[idx]], 2)
                                    
            else:
                track_object[ids[idx]] = [-1, num_frame, -1]

            text_conf = conf[idx]
            text_clss = clss[idx]
            text_ids = ids[idx]
            # text_inds = inds[idx]
            if text_clss == 0:
                num_car += 1
            if text_clss == 1:
                num_bike += 1
            if text_clss == 2:
                num_bus += 1
            if text_clss == 3:
                num_truck += 1
            
            label = "{}".format(text_ids)
            cv2.rectangle(img, (left, top), (right, bottom), colors[text_clss], 2)
            # cv2.putText(img, label, (left + 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[text_clss], 2)

    cv2.putText(img, 'num_car:{}'.format(num_car), (630, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[0], 2)
    cv2.putText(img, 'num_bike:{}'.format(num_bike), (630, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[1], 2)
    # cv2.putText(img, 'num_bus:{}'.format(num_bus), (600, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[2], 2)
    # cv2.putText(img, 'num_truck:{}'.format(num_truck), (600, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[3], 2)    



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


def get_roi_points(touch_angle, h_angle, real_height, left_real_width, right_real_width, width, height, diameter, ppa):
    width_angle = h_angle * math.pi / 180
    point_left = get_pixel(left_real_width, ppa, width_angle, touch_angle, real_height, height, width, diameter, True)
    point_right = get_pixel(right_real_width, ppa, width_angle, touch_angle, real_height, height, width, diameter,
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
    diameter = math.tan((math.pi / 180) * top) * real_height - math.tan((math.pi / 180) * bottom) * real_height
    return lst, diameter


def read_ccfg(height, width, diameter):
    file_path = FLAGS.ccfg
    file_path = './cfg/cfg.txt'

    print('use camera config file: ' + file_path)
    with open(file_path, 'r') as f:
      data = [float(i.strip()) for i in f.readlines()]

    real_height = data[0]
    left_width = data[1]
    right_width = data[2]
    base_angle = data[3]
    delta_angle = data[4]
    h_angle = data[5]
    top_rank = data[6]
    ppa = 0.5*height/delta_angle
    app = 2*delta_angle/height
    touch_angle = abs(base_angle-delta_angle)
    roi_points, diameter = get_roi_points(touch_angle, h_angle, real_height, left_width, right_width, width, height, diameter, ppa)
    left_area = diameter*left_width
    right_area = diameter*right_width

    return lambda x: math.tan((touch_angle+app*(height-x))*math.pi/180)*real_height, roi_points, left_area, right_area


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


def main(_argv):

    video_path = '/home/minhthanh/Downloads/a7_non_hd_front_normal_23s.mp4'
    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global fps
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    conf_threshold = 0.25
    tracking_class = [0, 1, 2, 3] # 0:car, 1:bike, 2:bus, 3:truck
    global colors
    global class_names
    colors = [(255, 144, 30), (0, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    with open("yolov5/data_txt/classes.names") as f:
        class_names = f.read().strip().split('\n')

    diameter = 30
    distance_cal, roi_points, left_area, right_area = read_ccfg(height, width, diameter)
    roi_points, raw_rois = region_of_interested(roi_points, width, height)
    top, bottom, right, left = search_pivot(roi_points)
    global roi_polygon
    roi_polygon = Polygon(roi_points)

    # save infor object: (distane, frame, speed)
    global track_object
    track_object = dict()

    global speed_function
    speed_function = speed_estimation(fps=fps)
    num_frame = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # for index in range(len(roi_points) - 1):
        #     cv2.line(frame, roi_points[index], roi_points[index + 1], (0, 0, 255), 2)

        pts = np.array(roi_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # inference detection
        results = model(frame)
        data = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        json_data = json.loads(data)

        if not json_data:
            continue

        # output has to be N X (x, y, x, y, conf, cls)
        num_frame += 1
        detect = []
        for record in json_data:
            confidence, class_id = round(record['confidence'], 2), record['class']
            left, top, right, bottom = int(record['xmin']), int(record['ymin']), int(record['xmax']), int(record['ymax'])
            # cv2.rectangle(frame, (left, top), (right, bottom), colors[class_id], 2)
            # cv2.putText(frame, "{}_{}".format(class_names[class_id], confidence), 
            #             (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
            if class_id in tracking_class and confidence >= conf_threshold:
                detect.append([left, top, right, bottom, confidence, class_id])

        # Check if there are any detections
        detect = np.array(detect)
        if len(detect) > 0:
            tracks = tracker.update(detect, frame) # --> M X (x, y, x, y, id, conf, cls, ind)
        # If no detections, make prediction ahead
        else:
            detect = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
            tracks = tracker.update(detect, frame) # --> M X (x, y, x, y, id, conf, cls, ind)

        # tracker.plot_results(im, show_trajectories=True) # auto draw object
        if (tracks is not None) and (len(tracks)>0):
            draw_box(frame, tracks, distance_cal, num_frame)

        # break on pressing q or space
        cv2.imshow('BoxMOT detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass