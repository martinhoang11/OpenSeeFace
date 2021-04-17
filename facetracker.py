import copy
import os
import sys
import argparse
import traceback
import gc
import math
from math import cos, sin
from scipy.spatial import distance as dist
import imutils
import pandas as pd
from rt_gene.estimate_gaze_tensorflow import GazeEstimator
from rt_gene.tracker_generic import GenericTracker
import matplotlib.pyplot as plt
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
parser.add_argument("--video-scale", type=int, help="This is a resolution scale factor applied to the saved AVI file", default=1, choices=[1,2,3,4])
parser.add_argument("--video-fps", type=float, help="This sets the frame rate of the output AVI file", default=24)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--ensamble", type=int, help="Set to 1 to use ensamble models(4 models)", default=0)
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--eye-gaze", type=int, help="When 1 eye gaze estimation will be calculated", default=0)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)
if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

class OutputLog(object):
    def __init__(self, fh, output):
        self.fh = fh
        self.output = output
    def write(self, buf):
        if not self.fh is None:
            self.fh.write(buf)
        self.output.write(buf)
        self.flush()
    def flush(self):
        if not self.fh is None:
            self.fh.flush()
        self.output.flush()

#define functions
def is_between(a, x, b):
    return min(a, b) < x < max(a, b)

output_logfile = None
if args.log_output != "":
    output_logfile = open(args.log_output, "w")
sys.stdout = OutputLog(output_logfile, sys.stdout)
sys.stderr = OutputLog(output_logfile, sys.stderr)

if os.name == 'nt':
    import dshowcapture
    if args.blackmagic == 1:
        dshowcapture.set_bm_enabled(True)
    if not args.blackmagic_options is None:
        dshowcapture.set_options(args.blackmagic_options)
    if not args.priority is None:
        import psutil
        classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
        p = psutil.Process(os.getpid())
        p.nice(classes[args.priority])

if os.name == 'nt' and (args.list_cameras > 0 or not args.list_dcaps is None):
    cap = dshowcapture.DShowCapture()
    info = cap.get_info()
    unit = 10000000.;
    if not args.list_dcaps is None:
        formats = {0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420", 201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2", 302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264" }
        for cam in info:
            if args.list_dcaps == -1:
                type = ""
                if cam['type'] == "Blackmagic":
                    type = "Blackmagic: "
                print(f"{cam['index']}: {type}{cam['name']}")
            if args.list_dcaps != -1 and args.list_dcaps != cam['index']:
                continue
            for caps in cam['caps']:
                format = caps['format']
                if caps['format'] in formats:
                    format = formats[caps['format']]
                if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
                else:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
    else:
        if args.list_cameras == 1:
            print("Available cameras:")
        for cam in info:
            type = ""
            if cam['type'] == "Blackmagic":
                type = "Blackmagic: "
            if args.list_cameras == 1:
                print(f"{cam['index']}: {type}{cam['name']}")
            else:
                print(f"{type}{cam['name']}")
    cap.destroy_capture()
    sys.exit(0)


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y

def visualize_eye_result(eye_image, est_gaze, tdx, tdy, center_x, center_y):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        # output_image = np.copy(eye_image)


        endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 50)

        # print(endpoint_x, endpoint_y)
        cv2.line(eye_image, (int(tdx), int(tdy)), (int(endpoint_x), int(endpoint_y)), (0, 255, 0))
        return eye_image

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def swap_xy(corr):
    arr = copy.deepcopy(corr)

    arr[0], arr[1] = arr[1], arr[0]
    return arr

def moving_av(mylist, N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves

def get_eye_area(img, eye_lms, name=None, visualize=False):
    x_min = int(min(x[1] for x in eye_lms)) - 5
    x_max = max(x[1] for x in eye_lms)
    y_min = int(min(x[0] for x in eye_lms)) - 5
    y_max = max(x[0] for x in eye_lms)

    h = int(y_max - y_min) + 5
    w = int(x_max - x_min)

    # cv2.rectangle(img, (x_min, y_min), (x_min+w, y_min+h), (0, 255, 0), 2)
    roi = img[y_min:y_min + h, x_min:x_min + w]
    # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    if visualize == True:
        cv2.imshow(name, roi)

    roi = cv2.resize(roi, (60,36), interpolation=cv2.INTER_CUBIC)
    return np.array(roi)

def input_from_image(cv_image):
    """This method converts an eye_img_msg provided by the landmark estimator, and converts it to a format
    suitable for the gaze network."""
    currimg = cv_image.reshape(36, 60, 3, order='F')
    currimg = currimg.astype(np.float32)
    testimg = np.zeros((36, 60, 3))
    testimg[:, :, 0] = currimg[:, :, 0] - 103.939
    testimg[:, :, 1] = currimg[:, :, 1] - 116.779
    testimg[:, :, 2] = currimg[:, :, 2] - 123.68
    return testimg

def visualize_eye_result(eye_image, est_gaze):
    """Here, we take the original eye eye_image and overlay the estimated gaze."""
    output_image = np.copy(eye_image)

    center_x = output_image.shape[1] / 2
    center_y = output_image.shape[0] / 2

    endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 50)

    cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0))
    return output_image

def get_normalised_eye_landmarks(landmarks, box):
    eye_indices = np.array([36, 39, 42, 45])
    transformed_landmarks = landmarks[eye_indices]
    transformed_landmarks[:, 0] -= box[0]
    transformed_landmarks[:, 1] -= box[1]
    return transformed_landmarks

def get_eye_image_from_landmarks(face_bb, face_img, landmarks, eye_image_size):
    eye_landmarks = get_normalised_eye_landmarks(landmarks, face_bb)
    margin_ratio = 1.0
    desired_ratio = float(eye_image_size[1]) / float(eye_image_size[0]) / 2.0

    try:
    # Get the width of the eye, and compute how big the margin should be according to the width
        lefteye_width = eye_landmarks[3][0] - eye_landmarks[2][0]
        righteye_width = eye_landmarks[1][0] - eye_landmarks[0][0]

        lefteye_center_x = eye_landmarks[2][0] + lefteye_width / 2
        righteye_center_x = eye_landmarks[0][0] + righteye_width / 2
        lefteye_center_y = (eye_landmarks[2][1] + eye_landmarks[3][1]) / 2.0
        righteye_center_y = (eye_landmarks[1][1] + eye_landmarks[0][1]) / 2.0

        aligned_face, rot_matrix = GenericTracker.align_face_to_eyes(face_img, right_eye_center=(righteye_center_x, righteye_center_y),
                                                                        left_eye_center=(lefteye_center_x, lefteye_center_y))
        # rotate the eye landmarks by same affine rotation to extract the correct landmarks
        ones = np.ones(shape=(len(eye_landmarks), 1))
        # points_ones = np.hstack([eye_landmarks, ones])
        transformed_eye_landmarks = rot_matrix.dot(eye_landmarks.T).T

        # recompute widths, margins and centers
        lefteye_width = transformed_eye_landmarks[3][0] - transformed_eye_landmarks[2][0]
        righteye_width = transformed_eye_landmarks[1][0] - transformed_eye_landmarks[0][0]
        lefteye_margin, righteye_margin = lefteye_width * margin_ratio, righteye_width * margin_ratio
        lefteye_center_y = (transformed_eye_landmarks[2][1] + transformed_eye_landmarks[3][1]) / 2.0
        righteye_center_y = (transformed_eye_landmarks[1][1] + transformed_eye_landmarks[0][1]) / 2.0

        # Now compute the bounding boxes
        # The left / right x-coordinates are computed as the landmark position plus/minus the margin
        # The bottom / top y-coordinates are computed according to the desired ratio, as the width of the image is known
        left_bb = np.zeros(4, dtype=np.int)
        left_bb[0] = transformed_eye_landmarks[2][0] - lefteye_margin / 2.0
        left_bb[1] = lefteye_center_y - (lefteye_width + lefteye_margin) * desired_ratio
        left_bb[2] = transformed_eye_landmarks[3][0] + lefteye_margin / 2.0
        left_bb[3] = lefteye_center_y + (lefteye_width + lefteye_margin) * desired_ratio

        right_bb = np.zeros(4, dtype=np.int)
        right_bb[0] = transformed_eye_landmarks[0][0] - righteye_margin / 2.0
        right_bb[1] = righteye_center_y - (righteye_width + righteye_margin) * desired_ratio
        right_bb[2] = transformed_eye_landmarks[1][0] + righteye_margin / 2.0
        right_bb[3] = righteye_center_y + (righteye_width + righteye_margin) * desired_ratio

        # Extract the eye images from the aligned image
        left_eye_color = aligned_face[left_bb[1]:left_bb[3], left_bb[0]:left_bb[2], :]
        right_eye_color = aligned_face[right_bb[1]:right_bb[3], right_bb[0]:right_bb[2], :]

        # So far, we have only ensured that the ratio is correct. Now, resize it to the desired size.
        left_eye_color_resized = cv2.resize(left_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)
        right_eye_color_resized = cv2.resize(right_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)

        return left_eye_color_resized, right_eye_color_resized, left_bb, right_bb
    except (ValueError, TypeError, cv2.error) as e:
        print('ERROR')
        return None, None, None, None

def get_image_from_bb(img, bb):
    bb = list(map(int, bb))
    roi = img[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
    return np.array(roi)

import numpy as np
import time
import cv2
import socket
import struct
import json
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path

if args.benchmark > 0:
    model_base_path = get_model_base_path(args.model_dir)
    im = cv2.imread(os.path.join(model_base_path, "benchmark.bin"), cv2.IMREAD_COLOR)
    results = []
    for model_type in [3, 2, 1, 0, -1, -2, -3]:
        tracker = Tracker(224, 224, threshold=0.1, max_threads=args.max_threads, max_faces=1, discard_after=0, scan_every=0, silent=True, model_type=model_type, model_dir=args.model_dir, no_gaze=(model_type == -1), detection_threshold=0.1, use_retinaface=0, max_feature_updates=900, static_model=True if args.no_3d_adapt == 1 else False)
        tracker.detected = 1
        tracker.faces = [(0, 0, 224, 224)]
        total = 0.0
        for i in range(100):
            start = time.perf_counter()
            r, rmat = tracker.predict(im)
            # r = tracker.predict(im)
            total += time.perf_counter() - start
        print(1. / (total / 100.))
    sys.exit(0)

target_ip = args.ip
target_port = args.port

if args.faces >= 40:
    print("Transmission of tracking data over network is not supported with 40 or more faces.")

fps = 24
dcap = None
use_dshowcapture_flag = False
if os.name == 'nt':
    fps = args.fps
    dcap = args.dcap
    use_dshowcapture_flag = True if args.use_dshowcapture == 1 else False
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
    if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
        fps = min(fps, input_reader.device.get_fps())
else:
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag)

if type(input_reader.reader) == VideoReader:
    fps = 0.0

log = None
out = None
first = True
height = 0
width = 0
tracker = None
sock = None
total_tracking_time = 0.0
tracking_time = 0.0
tracking_frames = 0
frame_count = 0
eye_blink_frames = 0
eye_blink_lst = []
eye_blink_temp = []
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1
COUNTER = 0
COUNTER_ORIGIN = 0


array_blink_threshold = list()
ear_list = list()
col=['F1',"F2","F3","F4","F5",'F6',"F7", "F8", "F9", "F10", "F11", "F12", "F13"]

features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

if args.log_data != "":
    log = open(args.log_data, "w")
    log.write("Frame,Time,Width,Height,FPS,Face,FaceID,RightOpen,LeftOpen,AverageConfidence,Success3D,PnPError,RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,Euler.X,Euler.Y,Euler.Z,RVec.X,RVec.Y,RVec.Z,TVec.X,TVec.Y,TVec.Z")
    for i in range(66):
        log.write(f",Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence")
    for i in range(66):
        log.write(f",Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z")
    for feature in features:
        log.write(f",{feature}")
    log.write("\r\n")
    log.flush()

is_camera = args.capture == str(try_int(args.capture))

###
if args.eye_gaze == 1:
    if args.ensamble == 1:
        gaze_estimator = GazeEstimator("/cpu:0", ['C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\rt_gene\\rt_gene\\model_nets\\all_subjects_mpii_prl_utmv_0_02.h5',
                                                'C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\rt_gene\\rt_gene\\model_nets\\all_subjects_mpii_prl_utmv_1_02.h5',
                                                'C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\rt_gene\\rt_gene\\model_nets\\all_subjects_mpii_prl_utmv_2_02.h5',
                                                'C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\rt_gene\\rt_gene\\model_nets\\all_subjects_mpii_prl_utmv_3_02.h5'])
    else:
        gaze_estimator = GazeEstimator("/cpu:0", 'C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\rt_gene\\rt_gene\\model_nets\\Model_allsubjects1.h5')
###

try:
    attempt = 0
    frame_time = time.perf_counter()
    target_duration = 0
    if fps > 0:
        target_duration = 1. / float(fps)
    repeat = args.repeat_video != 0 and type(input_reader.reader) == VideoReader
    need_reinit = 0
    failures = 0
    source_name = input_reader.name
    blink_count = 0
    blink_count_origin = 0
    while repeat or input_reader.is_open():
        if not input_reader.is_open() or need_reinit == 1:
            input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
            if input_reader.name != source_name:
                print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                sys.exit(1)
            need_reinit = 2
            time.sleep(0.02)
            continue
        if not input_reader.is_ready():
            time.sleep(0.02)
            continue
        
        ret, frame = input_reader.read()
        eye_gaze_frame = copy.deepcopy(frame)
        # frame = cv2.flip(frame,1)
        #2 -50 - 0.5 -20,-50
        # frame = cv2.convertScaleAbs(frame, -1, 0.5, -20)

        if not ret:
            if repeat:
                if need_reinit == 0:
                    need_reinit = 1
                continue
            elif is_camera:
                attempt += 1
                if attempt > 30:
                    break
                else:
                    time.sleep(0.02)
                    if attempt == 3:
                        need_reinit = 1
                    continue
            else:
                break;

        attempt = 0
        need_reinit = 0
        # frame_count += 1
        now = time.time()

        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
            if not args.video_out is None:
                out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('F','F','V','1'), args.video_fps, (width * args.video_scale, height * args.video_scale))

        try:
            inference_start = time.perf_counter()
            faces, rmat = tracker.predict(frame)
            # faces = tracker.predict(frame)
            if len(faces) > 0:
                inference_time = (time.perf_counter() - inference_start)
                total_tracking_time += inference_time
                tracking_time += inference_time / len(faces)
                tracking_frames += 1
            # else:
            #     ear_list.append(np.nan)
            #     array_blink_threshold.append(np.nan)
            packet = bytearray()
            detected = False
            r_eye_roi_resize = []
            l_eye_roi_resize = []
            head_list = []

            for face_num, f in enumerate(faces):
                f = copy.copy(f)
                f.id += args.face_id_offset
                if f.eye_blink is None:
                    f.eye_blink = [1, 1]

                right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                left_state = "O" if f.eye_blink[1] > 0.30 else "-"
                if f.eye_blink[0] < 0.7 or f.eye_blink[1] < 0.7:
                    eye_blink_frames += 1

                if args.silent == 0:
                    print(f"Confidence[{f.id}]: {f.conf:.4f} / 3D fitting error: {f.pnp_error:.4f} / Eyes: {left_state}, {right_state}")
                
                detected = True
                if not f.success:
                    pts_3d = np.zeros((70, 3), np.float32)
                packet.extend(bytearray(struct.pack("d", now)))
                packet.extend(bytearray(struct.pack("i", f.id)))
                packet.extend(bytearray(struct.pack("f", width)))
                packet.extend(bytearray(struct.pack("f", height)))
                packet.extend(bytearray(struct.pack("f", f.eye_blink[0])))
                packet.extend(bytearray(struct.pack("f", f.eye_blink[1])))
                packet.extend(bytearray(struct.pack("B", 1 if f.success else 0)))
                packet.extend(bytearray(struct.pack("f", f.pnp_error)))
                packet.extend(bytearray(struct.pack("f", f.quaternion[0])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[1])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[2])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[3])))
                packet.extend(bytearray(struct.pack("f", f.euler[0])))
                packet.extend(bytearray(struct.pack("f", f.euler[1])))
                packet.extend(bytearray(struct.pack("f", f.euler[2])))
                packet.extend(bytearray(struct.pack("f", f.translation[0])))
                packet.extend(bytearray(struct.pack("f", f.translation[1])))
                packet.extend(bytearray(struct.pack("f", f.translation[2])))

                ###mouth opening - Thresh hold
                try:
                    if f.current_features['mouth_open'] > 0.5 and f.current_features['mouth_wide'] < 0.1:
                        frame = cv2.putText(frame, 'Open', (80,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
                except:
                    pass

                if is_between(0, f.euler[0], 150):
                    pitch_stt = 'up'
                elif f.euler[0] < 0:
                    pitch_stt = 'down'
                else:
                    pitch_stt = 'straight'

                if f.euler[1] > 30:
                    yaw_stt = 'left'
                elif f.euler[1] < 0:
                    yaw_stt = 'right'
                else:
                    yaw_stt = 'straight'
                
                if f.euler[2] > 110:
                    roll_stt = 'right'
                elif f.euler[2] < 65:
                    roll_stt = 'left'
                else:
                    roll_stt = 'straight'


                ###

                ###
                if f.euler[0] > 0:
                    pitch = f.euler[0] - 170
                else:
                    pitch = f.euler[0] + 170
                yaw = f.euler[1] - 18
                roll = f.euler[2] - 86
                ###

                ###eye blink
                # if eye_blink_frames >= 3:
                #     blink_count += 1
                #     eye_blink_frames = 0
                ####

                ###eye gaze
                pupil_r_x = int((f.lms[36][1] + f.lms[39][1]) / 2)
                pupil_r_y = int((f.lms[36][0] + f.lms[39][0]) / 2)
                pupil_l_x = int((f.lms[42][1] + f.lms[45][1]) / 2)
                pupil_l_y = int((f.lms[42][0] + f.lms[45][0]) / 2)

                right_gaze = (f.pts_3d[66] - f.pts_3d[68])*100
                left_gaze = (f.pts_3d[67] - f.pts_3d[69])*100
                
                if right_gaze[0] > 2 and left_gaze[0] > 0:
                    eye_stt_lr = 'right'
                    # cv2.arrowedLine(frame, (pupil_l_x, pupil_l_y), (int(f.lms[67][1])-40, int(f.lms[67][0])), (0,255,255), 2, tipLength=0.2) #GREEN
                    # cv2.arrowedLine(frame, (pupil_r_x, pupil_r_y), (int(f.lms[66][1])-40, int(f.lms[66][0])), (0,255,255), 2, tipLength=0.2) #GREEN

                elif left_gaze[0] <-2 and right_gaze[0] < -1.5  or left_gaze[0] < -4:
                    eye_stt_lr = 'left'
                    # cv2.arrowedLine(frame, (pupil_l_x, pupil_l_y), (int(f.lms[67][1])+40, int(f.lms[67][0])), (0,255,255), 2, tipLength=0.2) #GREEN
                    # cv2.arrowedLine(frame, (pupil_r_x, pupil_r_y), (int(f.lms[66][1])+40, int(f.lms[66][0])), (0,255,255), 2, tipLength=0.2) #GREEN

                else:
                    eye_stt_lr = 'straight'
                
                if right_gaze[1] > 2.5 or left_gaze[1] > 2.5:
                    eye_stt_ud = 'up'
                    # cv2.arrowedLine(frame, (pupil_l_x, pupil_l_y), (int(f.lms[67][1]), int(f.lms[67][0])-40), (0,255,255), 2, tipLength=0.2) #GREEN
                    # cv2.arrowedLine(frame, (pupil_r_x, pupil_r_y), (int(f.lms[66][1]), int(f.lms[66][0])-40), (0,255,255), 2, tipLength=0.2) #GREEN

                elif right_gaze[1] < 0 or left_gaze[1] < 0:
                    eye_stt_ud = 'down'
                    # cv2.arrowedLine(frame, (pupil_l_x, pupil_l_y), (int(f.lms[67][1]), int(f.lms[67][0])+40), (0,255,255), 2, tipLength=0.2) #GREEN
                    # cv2.arrowedLine(frame, (pupil_r_x, pupil_r_y), (int(f.lms[66][1]), int(f.lms[66][0])+40), (0,255,255), 2, tipLength=0.2) #GREEN
                else:
                    eye_stt_ud = 'straight'

                # visualize_eye_result(frame, (f.lms[66][1]*100, f.lms[66][0]*100), tdx=f.lms[66][1], tdy=f.lms[66][0], center_x=pupil_r_x, center_y=pupil_r_y)
                # visualize_eye_result(frame, (f.lms[67][1]*100, f.lms[67][0]*100), tdx=f.lms[67][1], tdy=f.lms[67][0], center_x=pupil_l_x, center_y=pupil_l_y)
                #### head pose visualize
               
                draw_axis(frame, -f.euler[1]+17, f.euler[0]+10, f.euler[2]+3, tdx=f.lms[30][1], tdy=f.lms[30][0], size = 100)
                #### save ear to csv
                # if len(eye_blink_temp) == 13:
                #     eye_blink_lst.append(eye_blink_temp)
                #     eye_blink_temp = []
                # else:
                #     eye_blink_temp.append(np.average(f.eye_blink))

                r_eye = [swap_xy(f.lms[36]), swap_xy(f.lms[37]), swap_xy(f.lms[38]), swap_xy(f.lms[39]), swap_xy(f.lms[40]), swap_xy(f.lms[41])]
                l_eye = [swap_xy(f.lms[42]), swap_xy(f.lms[43]), swap_xy(f.lms[44]), swap_xy(f.lms[45]), swap_xy(f.lms[46]), swap_xy(f.lms[47])]
                ear_origin = np.average((eye_aspect_ratio(r_eye), eye_aspect_ratio(l_eye)))


                ear = np.average(f.eye_blink)
                # # ear_list.append(ear)
                # # array_blink_threshold.append(0) 

                if ear < 0.7:
                    COUNTER += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        blink_count += 1
                    # reset the eye frame counter
                    COUNTER = 0
                
                if ear_origin < 0.19:
                    COUNTER_ORIGIN += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold`
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if COUNTER_ORIGIN >= EYE_AR_CONSEC_FRAMES:
                        blink_count_origin += 1
                    # reset the eye frame counter
                    COUNTER_ORIGIN = 0

                ######
                if args.eye_gaze == 1:
                    _rotation_matrix = np.matmul(rmat, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
                    _m = np.zeros((4, 4))
                    _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
                    _m[:3, :3] = _rotation_matrix
                    _m[3, 3] = 1
                    # Go from camera space to ROS space
                    _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.0],
                                    [0.0, -1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]

                    roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
                    roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

                    phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)


                    # le_c, re_c, _, _ = get_eye_image_from_landmarks(f.bbox, get_image_from_bb(frame, f.bbox), f.lms, (60,36))
                    re_c = get_eye_area(eye_gaze_frame, [f.lms[36], f.lms[37], f.lms[38], f.lms[39], f.lms[40], f.lms[41]])
                    le_c = get_eye_area(eye_gaze_frame, [f.lms[42], f.lms[43], f.lms[44], f.lms[45], f.lms[46], f.lms[47]])

                    r_eye_roi_resize.append(input_from_image(re_c))
                    l_eye_roi_resize.append(input_from_image(le_c))
                    head_list.append([phi_head, theta_head])

                #########
                frame_count += 1
                if not log is None:
                    log.write(f"{frame_count},{now},{width},{height},{args.fps},{face_num},{f.id},{f.eye_blink[0]},{f.eye_blink[1]},{f.conf},{f.success},{f.pnp_error},{f.quaternion[0]},{f.quaternion[1]},{f.quaternion[2]},{f.quaternion[3]},{f.euler[0]},{f.euler[1]},{f.euler[2]},{f.rotation[0]},{f.rotation[1]},{f.rotation[2]},{f.translation[0]},{f.translation[1]},{f.translation[2]}")
                for (x,y,c) in f.lms:
                    packet.extend(bytearray(struct.pack("f", c)))
                if args.visualize > 1:
                    frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))
                    frame = cv2.putText(frame, "FPS : %0.1f" % fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

                    frame = cv2.putText(frame, 'blink: (model)' + str(blink_count) + ' - (ear)' + str(blink_count_origin), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (int(f.bbox[0]),int(f.bbox[1])), (int(f.bbox[0]+f.bbox[2]),int(f.bbox[1]+f.bbox[3])), (0,0,255), 1)
                    frame = cv2.putText(frame, 'mouth: ', (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"h_pitch: {pitch_stt}({round(pitch, 3)})", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"h_yaw: {yaw_stt}({round(yaw, 3)})", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"h_roll: {roll_stt}({round(roll, 3)})", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"eye_ud: {eye_stt_ud}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"eye_lr: {eye_stt_lr}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"eye_x_lr: {left_gaze[0]:.3f}/{right_gaze[0]:.3f}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"eye_y_ud: {left_gaze[1]:.3f}/{right_gaze[1]:.3f}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)
                    frame = cv2.putText(frame, f"EAR: {f.eye_blink[0]:.3f}-{f.eye_blink[1]:.3f}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 1, cv2.LINE_AA)

                if args.visualize > 2:
                    frame = cv2.putText(frame, f"{f.conf:.4f}", (int(f.bbox[0] + 18), int(f.bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                for pt_num, (x,y,c) in enumerate(f.lms):
                    packet.extend(bytearray(struct.pack("f", y)))
                    packet.extend(bytearray(struct.pack("f", x)))
                    if not log is None:
                        log.write(f",{y},{x},{c}")
                    if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.30):
                        continue
                    if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.30):
                        continue
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                    if args.visualize != 0 or not out is None:
                        if args.visualize > 3:
                            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                        color = (0, 255, 0)
                        if pt_num >= 66:
                            color = (255, 255, 0)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                        x += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                        y += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                        x -= 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                if args.pnp_points != 0 and (args.visualize != 0 or not out is None) and f.rotation is not None:
                    if args.pnp_points > 1:
                        projected = cv2.projectPoints(f.face_3d[0:66], f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    else:
                        
                        projected = cv2.projectPoints(f.contour, f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    for [(x,y)] in projected[0]:
                        x = int(x + 0.5)
                        y = int(y + 0.5)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        y += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x -= 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                for (x,y,z) in f.pts_3d:
                    packet.extend(bytearray(struct.pack("f", x)))
                    packet.extend(bytearray(struct.pack("f", -y)))
                    packet.extend(bytearray(struct.pack("f", -z)))
                    if not log is None:
                        log.write(f",{x},{-y},{-z}")
                if f.current_features is None:
                    f.current_features = {}

                for feature in features:
                    if not feature in f.current_features:
                        f.current_features[feature] = 0
                    packet.extend(bytearray(struct.pack("f", f.current_features[feature])))
                    if not log is None:
                        log.write(f",{f.current_features[feature]}")
                if not log is None:
                    log.write("\r\n")
                    log.flush()

            ##########eye gaze
            if args.eye_gaze == 1:
                gaze_est = gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=l_eye_roi_resize,
                                                        inference_input_right_list=r_eye_roi_resize,
                                                        inference_headpose_list=head_list)
                
                for gaze, headpose in zip(gaze_est.tolist(), head_list):
                    # Build visualizations
                    r_gaze_img = gaze_estimator.visualize_eye_result(re_c, gaze)
                    l_gaze_img = gaze_estimator.visualize_eye_result(le_c, gaze)
                    s_gaze_img = np.concatenate((cv2.resize(r_gaze_img, (112,112)), cv2.resize(l_gaze_img, (112,112))), axis=1)
                    
                    cv2.imshow('eye_gaze', s_gaze_img)
    
            #########3
            if detected and len(faces) < 40:
                sock.sendto(packet, (target_ip, target_port))

            if not out is None:
                video_frame = frame
                if args.video_scale != 1:
                    video_frame = cv2.resize(frame, (width * args.video_scale, height * args.video_scale), interpolation=cv2.INTER_NEAREST)
                out.write(video_frame)
                if args.video_scale != 1:
                    del video_frame

            if args.visualize != 0:
                cv2.imshow('OpenSeeFace Visualization', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if args.dump_points != "" and not faces is None and len(faces) > 0:
                        np.set_printoptions(threshold=sys.maxsize, precision=15)
                        pairs = [
                            (0, 16),
                            (1, 15),
                            (2, 14),
                            (3, 13),
                            (4, 12),
                            (5, 11),
                            (6, 10),
                            (7, 9),
                            (17, 26),
                            (18, 25),
                            (19, 24),
                            (20, 23),
                            (21, 22),
                            (31, 35),
                            (32, 34),
                            (36, 45),
                            (37, 44),
                            (38, 43),
                            (39, 42),
                            (40, 47),
                            (41, 46),
                            (48, 52),
                            (49, 51),
                            (56, 54),
                            (57, 53),
                            (58, 62),
                            (59, 61),
                            (65, 63)
                        ]
                        points = copy.copy(faces[0].face_3d)
                        for a, b in pairs:
                            x = (points[a, 0] - points[b, 0]) / 2.0
                            y = (points[a, 1] + points[b, 1]) / 2.0
                            z = (points[a, 2] + points[b, 2]) / 2.0
                            points[a, 0] = x
                            points[b, 0] = -x
                            points[[a, b], 1] = y
                            points[[a, b], 2] = z
                        points[[8, 27, 28, 29, 33, 50, 55, 60, 64], 0] = 0.0
                        points[30, :] = 0.0
                        with open(args.dump_points, "w") as fh:
                            fh.write(repr(points))
                    break
            failures = 0
        except Exception as e:
            if e.__class__ == KeyboardInterrupt:
                if args.silent == 0:
                    print("Quitting")
                break
            traceback.print_exc()
            failures += 1
            if failures > 30:
                break


        collected = False
        del frame

        duration = time.perf_counter() - frame_time
        while duration < target_duration:
            if not collected:
                gc.collect()
                collected = True
            duration = time.perf_counter() - frame_time
            sleep_time = target_duration - duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            duration = time.perf_counter() - frame_time
        frame_time = time.perf_counter()
        
except KeyboardInterrupt:
    if args.silent == 0:
        print("Quitting")

input_reader.close()
if not out is None:
    out.release()
cv2.destroyAllWindows()

if args.silent == 0 and tracking_frames > 0:
    average_tracking_time = 1000 * tracking_time / tracking_frames
    print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
    print(f"Tracking time: {total_tracking_time:.3f} s\nFrames: {tracking_frames}")





    # df = pd.DataFrame(eye_blink_lst, columns=col)
    # df.to_csv('test_model_4_13f.csv')
    # mov_ear_3=moving_av(ear_list,3)
    # mov_ear_5=moving_av(ear_list,5)
    # mov_ear_7=moving_av(ear_list,7)

    # ear_list = pd.Series(ear_list, index=range(0, len(ear_list)))
    # array_blink_threshold=pd.Series(array_blink_threshold,index=range(0, len(array_blink_threshold)))

    # mov_ear_3=pd.Series(mov_ear_3, index=range(2, len(mov_ear_3)+2))
    # mov_ear_5=pd.Series(mov_ear_5, index=range(3, len(mov_ear_5)+3))
    # mov_ear_7=pd.Series(mov_ear_7, index=range(4, len(mov_ear_7)+4))

    # ear_list = pd.DataFrame(ear_list)
    # ear_list["threshold"] = array_blink_threshold
    # ear_list["mov_ear_3"] = mov_ear_3
    # ear_list["mov_ear_5"] = mov_ear_5
    # ear_list["mov_ear_7"] = mov_ear_7
    # ear_list.columns = ["ear", "threshold", "mov_ear_3","mov_ear_5","mov_ear_7"]
    # #ear_list = ear_list.fillna(0)
    # #mask = ear_list.tag == 0
    # #ear_list.tag = ear_list.tag.where(mask, 1)

    # ear_list.index.name="frame"
    # '''
    # try:
    #     ear_list.to_csv("non_training_data_raw_data/{}/{}.csv".format(
    #             args["video"][6:-4]), index=True, header=True)
    # except FileNotFoundError:
    #     ear_list.to_csv("non_training_data_raw_data/{}.csv".format(
    #             args["video"][7:-4]), index=True, header=True)
    # # do a bit of cleanup
    # cv2.destroyAllWindows()
    # vs.stop()
    # '''
    # ear_list.to_csv("model_4.csv",index=True, header=True)

