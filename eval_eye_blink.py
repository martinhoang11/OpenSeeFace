import copy
import os
import sys
import argparse
import traceback
import gc
import numpy as np
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
from imutils import paths
import time
import h5py
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, roc_auc_score


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
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.02)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=1)
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

def swap_xy(corr):
    arr = copy.deepcopy(corr)

    arr[0], arr[1] = arr[1], arr[0]
    return arr

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def process_video(video_path):
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

    import numpy as np
    import time
    import cv2
    import socket
    import struct
    import json
    from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
    from tracker import Tracker, get_model_base_path
    from tqdm import tqdm

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
                r = tracker.predict(im)
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
        input_reader = InputReader(video_path, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
        if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
            fps = min(fps, input_reader.device.get_fps())
    else:
        input_reader = InputReader(video_path, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag)

    # if type(input_reader.reader) == VideoReader:
    #     fps = 0.0

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
    framecount = 0
    eye_blink_frames = 0
    eye_blink_lst = []
    eye_blink_temp = []

    COUNTER = 0
    TOTAL = 0
    current_frame = 1
    blink_start = 0
    blink_end = 0
    closeness = 0
    output_closeness = []
    output_blinks = []
    blink_info = (0,0)
    processed_frame = []
    frame_info_list = []
    lStart = 42
    lEnd = 48
    rStart = 36
    rEnd = 42
    ear_th = 0.18
    consec_th = 3
    up_to = None


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

    is_camera = video_path == str(try_int(video_path))

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
                input_reader = InputReader(video_path, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
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

            fps = input_reader.get_fps()
            frame_count = int(input_reader.get_frame())
            duration = frame_count/fps
            video_info_dict = {
                'fps': fps,
                'frame_count': frame_count,
                'duration(s)': duration
            }

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
                faces = tracker.predict(frame)
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

                    r_eye = [[f.lms[36][1], f.lms[36][0]], [f.lms[37][1], f.lms[37][0]], [f.lms[38][1], f.lms[38][0]], [f.lms[39][1], f.lms[39][0]], [f.lms[40][1], f.lms[40][0]], [f.lms[41][1], f.lms[41][0]]]
                    l_eye = [[f.lms[42][1], f.lms[42][0]], [f.lms[43][1], f.lms[43][0]], [f.lms[44][1], f.lms[44][0]], [f.lms[45][1], f.lms[45][0]], [f.lms[46][1], f.lms[46][0]], [f.lms[47][1], f.lms[47][0]]]
                    l_eye_ratio = eye_aspect_ratio(l_eye)
                    r_eye_ratio = eye_aspect_ratio(r_eye)
                    ear = (l_eye_ratio + r_eye_ratio) / 2.0
                    ear_model = (f.eye_blink[0] + f.eye_blink[1]) / 2.0

                    if ear_model < 0.7: #0.21
                        COUNTER += 1
                        closeness = 1
                        output_closeness.append(closeness)
                    else:
                        if COUNTER >= consec_th:
                            TOTAL += 1
                            blink_start = current_frame - COUNTER
                            blink_end = current_frame - 1
                            blink_info = (blink_start, blink_end)
                            output_blinks.append(blink_info)
                        COUNTER = 0
                        closeness = 0
                        output_closeness.append(closeness)
                    
                    frame_info = {
                        'frame_no': current_frame,
                        'face_detected': 1,
                        'face_coordinates': 0,
                        'left_eye_coor': 0,
                        'right_eye_coor': 0,
                        'left_ear': l_eye_ratio,
                        'right_ear': r_eye_ratio,
                        'avg_ear': ear,
                        'avg_ear_model': ear_model,
                        'closeness': closeness,
                        'blink_no': TOTAL,
                        'blink_start_frame': blink_start,
                        'blink_end_frame': blink_end,
                        'reserved_for_calibration': False
                    }
                    frame_info_list.append(frame_info)
                    processed_frame.append(frame)
                    current_frame += 1
                    # frame_info_df = pd.DataFrame(frame_info_list) # debug
                    
                    framecount += 1
                    if not log is None:
                        log.write(f"{framecount},{now},{width},{height},{args.fps},{face_num},{f.id},{f.eye_blink[0]},{f.eye_blink[1]},{f.conf},{f.success},{f.pnp_error},{f.quaternion[0]},{f.quaternion[1]},{f.quaternion[2]},{f.quaternion[3]},{f.euler[0]},{f.euler[1]},{f.euler[2]},{f.rotation[0]},{f.rotation[1]},{f.rotation[2]},{f.translation[0]},{f.translation[1]},{f.translation[2]}")
                    for (x,y,c) in f.lms:
                        packet.extend(bytearray(struct.pack("f", c)))
                    if args.visualize > 1:
                        frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))
                        frame = cv2.putText(frame, "FPS : %0.1f" % fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

                    if args.visualize > 2:
                        frame = cv2.putText(frame, f"{f.conf:.4f}", (int(f.bbox[0] + 18), int(f.bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                    
                    # cv2.imwrite('frame.jpg', frame)
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
                

                if detected and len(faces) < 40:
                    sock.sendto(packet, (target_ip, target_port))

                if not out is None:
                    video_frame = frame
                    if args.video_scale != 1:
                        video_frame = cv2.resize(frame, (width * args.video_scale, height * args.video_scale), interpolation=cv2.INTER_NEAREST)
                    out.write(video_frame)
                    if args.video_scale != 1:
                        del video_frame

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

    frame_info_df = pd.DataFrame(frame_info_list)
    frame_info_df['output_closeness'] = output_closeness

    file_name = os.path.basename(video_path)
    output_str = 'Processing {} has done.\n\n'.format(file_name)
    return frame_info_df, output_closeness, output_blinks, processed_frame, video_info_dict, output_str

def skip_first_n_frames(frame_info_df, closeness_list, blink_list, processed_frames, skip_n=0, consec_th=3):
    # recalculate closeness_list
    recalculated_closeness_list = closeness_list[skip_n:] # skip first n frames

    # update 'reserved_for_calibration' column of frame_info_df for first "skip_n" frames
    frame_info_df.loc[:skip_n-1, 'reserved_for_calibration'] = True # .loc includes second index -> [first:second] 
    
    # recalculate blink_list
    # get blink count in the first "SKIP_FIRST_FRAMES" frames
    blink_count_til_n = frame_info_df.loc[skip_n, 'blink_no']    
    # determine start of the blink that comes after first n frames
    start_of_blink = blink_list[blink_count_til_n][0] - 1   #(-1) since frame-codes in blink_list start from 1
    # if some frames of the the blink starts before n
    if start_of_blink < skip_n: 
        # find frames of the blink that comes before n
        frames_to_discard = skip_n - start_of_blink
        # find duration of the blink
        duration_of_blink = blink_list[blink_count_til_n][1] - blink_list[blink_count_til_n][0] + 1
        # calculate new duration of blink after discarding first n frames
        new_duration = duration_of_blink - frames_to_discard
        # if new duration of the blink that comes after first n frames is less than n 
        if new_duration < consec_th:
            # then reduce total blink count by (blink_count_til_n + 1)
            recalculated_blink_list = blink_list[blink_count_til_n + 1:]
        # if new duration of the blink is NOT less than n 
        else:
            # then reduce total blink count by (blink_count_til_n)
            recalculated_blink_list = blink_list[blink_count_til_n:]
    # if the blink starts after n
    else:
        # then reduce total blink count by (blink_count_til_n)
        recalculated_blink_list = blink_list[blink_count_til_n:]
            
    # re-assign the frame-codes of recalculated_blinks if some frames are discarded       
    if skip_n > 0: 
        recalculated_blink_list = [(blink[0]-skip_n, blink[1]-skip_n) for blink in recalculated_blink_list]
        
    # also discard first n frames of "processed_frames"
    recalculated_processed_frames = processed_frames[skip_n:]
    
    return frame_info_df, recalculated_closeness_list, recalculated_blink_list, recalculated_processed_frames

def display_stats(closeness_list, blinks_list, video_info = None, skip_n = 0, test = False):
    str_out = ""
    # write video info
    if video_info != None:
        str_out += ("Video info\n")
        str_out += ("FPS: {}\n".format(video_info["fps"]))
        str_out += ("FRAME_COUNT: {}\n".format(video_info["frame_count"]))
        str_out += ("DURATION (s): {:.2f}\n".format(video_info["duration(s)"]))
        str_out += ("\n")
    
    # if you skipped n frames previously
    if skip_n > 0:
        str_out += ("After skipping {} frames,\n".format(skip_n))   
        
    # if you are displaying prediction information
    if test == False:    
        str_out += ("Statistics on the prediction set are\n")
    
    # if you are displaying test information
    if test == True:    
        str_out += ("Statistics on the test set are\n")
    
    str_out += ("TOTAL NUMBER OF FRAMES PROCESSED: {}\n".format(len(closeness_list)))
    str_out += ("NUMBER OF CLOSED FRAMES: {}\n".format(closeness_list.count(1)))
    str_out += ("NUMBER OF BLINKS: {}\n".format(len(blinks_list)))
    str_out += ("\n")
    
    print(str_out)
    return str_out

def read_annotations(input_file, skip_n = 0):
    # define variables 
    blink_start = 1
    blink_end = 1
    blink_info = (0,0)
    blink_list = []
    closeness_list = []

    # Using readlines() 
    file1 = open(input_file) 
    Lines = file1.readlines() 

    # find "#start" line 
    start_line = 1
    for line in Lines: 
        clean_line=line.strip()
        if clean_line=="#start":
            break
        start_line += 1

    # convert tag file to readable format and build "closeness_list" and "blink_list"
    for index in range(len(Lines[start_line+skip_n : -1])): # -1 since last line will be"#end"
        
        # read previous annotation and current annotation 
        prev_annotation=Lines[start_line+skip_n+index-1].split(':')
        current_annotation=Lines[start_line+skip_n+index].split(':')
        
        # if previous annotation is not "#start" line and not "blink" and current annotation is a "blink"
        if prev_annotation[0] != "#start\n" and prev_annotation[1] == "-1" and int(current_annotation[1]) > 0:
            # it means a new blink starts so save frame id as starting frame of the blink
            blink_start = int(current_annotation[0])
        
        # if previous annotation is not "#start" line and is a "blink" and current annotation is not a "blink"
        if prev_annotation[0] != "#start\n" and int(prev_annotation[1]) > 0 and current_annotation[1] == "-1":
            # it means a new blink ends so save (frame id - 1) as ending frame of the blink
            blink_end = int(current_annotation[0]) - 1
            # and construct a "blink_info" tuple to append the "blink_list"
            blink_info = (blink_start,blink_end)
            blink_list.append(blink_info)
        
        # if current annotation consist fully closed eyes, append it also to "closeness_list" 
        if current_annotation[3] == "C" and current_annotation[5] == "C":
            closeness_list.append(1)
        
        else:
            closeness_list.append(0)
    
    file1.close()
    return closeness_list, blink_list

def display_test_scores(closeness_list_test, closeness_list_pred):
    str_out = ""
    str_out += ("EYE CLOSENESS FRAME BY FRAME TEST SCORES\n")
    str_out += ("\n")

    #print accuracy
    accuracy = accuracy_score(closeness_list_test, closeness_list_pred)
    str_out += ("ACCURACY: {:.4f}\n".format(accuracy))
    str_out += ("\n")

    #print AUC score
    auc = roc_auc_score(closeness_list_test, closeness_list_pred)
    str_out += ("AUC: {:.4f}\n".format(auc))
    str_out += ("\n")

    #print confusion matrix
    str_out += ("CONFUSION MATRIX:\n")
    conf_mat = confusion_matrix(closeness_list_test, closeness_list_pred)
    str_out += ("{}".format(conf_mat))
    str_out += ("\n")
    str_out += ("\n")

    #print FP, FN
    str_out += ("FALSE POSITIVES:\n")
    fp = conf_mat[1][0]
    pos_labels = conf_mat[1][0]+conf_mat[1][1]
    str_out += ("{} out of {} positive labels ({:.4f}%)\n".format(fp, pos_labels,fp/pos_labels))
    str_out += ("\n")

    str_out += ("FALSE NEGATIVES:\n")
    fn = conf_mat[0][1]
    neg_labels = conf_mat[0][1]+conf_mat[0][0]
    str_out += ("{} out of {} negative labels ({:.4f}%)\n".format(fn, neg_labels, fn/neg_labels))
    str_out += ("\n")

    #print classification report
    str_out += ("PRECISION, RECALL, F1 scores:\n")
    str_out += ("{}".format(classification_report(closeness_list_test, closeness_list_pred)))
    
    print(str_out)
    return str_out

def write_outputs(input_file_name, closeness_list, blinks_list, frame_info_df=None, scores=None, \
                  test=False, scores_only=False):
    # clean filename from path and extensions so you can pass input_file variable to function as it is.
    clean_filename=os.path.basename(os.path.splitext(input_file_name)[0])
    
    # if you are writing prediction outputs
    if test == False and scores_only == False:
        #write all lists to single .h5 file
        with h5py.File("data_training\\{}_pred.h5".format(clean_filename), "w") as hf:
            g = hf.create_group('pred')
            g.create_dataset('closeness_list',data=closeness_list)
            g.create_dataset('blinks_list',data=blinks_list)
            if frame_info_df is not None:
                frame_info_df.to_parquet('data_training\\{}_frame_info_df.parquet'.format(clean_filename), engine='pyarrow')
            
    # if you are writing test outputs
    if test == True and scores_only == False:
        #write all lists to single .h5 file
        with h5py.File("data_training\\{}_test.h5".format(clean_filename), "w") as hf:
            g = hf.create_group('test')
            g.create_dataset('closeness_list',data=closeness_list)
            g.create_dataset('blinks_list',data=blinks_list)
            if frame_info_df is not None:
                frame_info_df.to_parquet('data_training\\{}_frame_info_df.parquet'.format(clean_filename), engine='pyarrow')

   # if you are writing scores
    if scores != None:
        # use text files this time
        with open("{}_scores.txt".format(clean_filename),"w", encoding='utf-8') as f:
            f.write(scores)
    return

def read_outputs(h5_name, parquet_name=None, test=False):
    # read h5 file by name
    hf = h5py.File('{}.h5'.format(h5_name), 'r')
    
    # if you are reading prediction results
    if test == False:  
        g = hf.get("pred") # read group first   
        
    # if you are reading test results
    if test == True:
         g = hf.get("test") # read group first  
            
    # then get datasets
    closeness_list = list(g.get('closeness_list'))
    blink_list = list(g.get('blinks_list'))

    # if you want to read frame_df_info
    if parquet_name != None:
        frame_info_df = pd.read_parquet('{}.parquet'.format(parquet_name), engine='pyarrow')
        return closeness_list, blink_list, frame_info_df
    
    # if you don't want to read frame_df_info
    else:
        return closeness_list, blink_list

def load_datasets(path, dataset_name):              
    # build  full path
    full_path = os.path.join(path, dataset_name)
    
    # read prediction results and frame_info_df
    closeness_pred, blinks_pred, frame_info_df \
                = read_outputs("{}_pred".format(full_path),"{}_frame_info_df".format(full_path))

    # read test results
    closeness_test, blinks_test = read_outputs("{}_test".format(full_path), test = True)
    
    # read scores
    with open("{}_scores.txt".format(full_path),"r") as f:
        Lines = f.readlines() 
        # build a string that hold scores
        scores_str = ""
        for line in Lines: 
            scores_str += line

    return  closeness_pred, blinks_pred, frame_info_df, closeness_test, blinks_test, scores_str

if __name__ == '__main__':
    # from imutils import paths
    # avi_video_files = []
    # tag_video_files = []
    # for i in list(paths.list_files('.\\eye_blink'))[1:]:
    #     if i.endswith('.avi'):
    #         avi_video_files.append(i)
    #     elif i.endswith('.tag'):
    #         tag_video_files.append(i)
    SKIP_FIRST_FRAMES = 0
    # for avi_video in avi_video_files:
    #     print(f'Processing {avi_video}')
    file_path = "C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\eye_blink\\eyeblink8\\4\\26122013_230654_cam.tag"
    # file_path = "C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\eye_blink\\talkingFace\\talking.tag"
    avi_video = "C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\eye_blink\\eyeblink8\\4\\26122013_230654_cam.avi"
    # avi_video = "C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace\\eye_blink\\talkingFace\\talking.avi"
    frame_info_df, closeness_predictions, blink_predictions, frames, video_info, scores_string \
        = process_video(avi_video)

    frame_info_df, closeness_predictions_skipped, blink_predictions_skipped, frames_skipped \
        = skip_first_n_frames(frame_info_df, closeness_predictions, blink_predictions, frames, \
            skip_n = SKIP_FIRST_FRAMES)

    frame_info_df.to_csv(f'{os.path.join("data_training", os.path.basename(avi_video))}.csv')
     

    scores_string += display_stats(closeness_predictions, blink_predictions, video_info)

    # # then display statistics by using outputs of skip_first_n_frames() function which are 
    # #"closeness_predictions_skipped" and "blinks_predictions_skipped"
    # if(SKIP_FIRST_FRAMES > 0):
    #     scores_string += display_stats(closeness_predictions_skipped, blink_predictions_skipped, video_info, \
    #                             skip_n = SKIP_FIRST_FRAMES)
    

    # read tag file
    closeness_test, blinks_test = read_annotations(file_path, skip_n = SKIP_FIRST_FRAMES)

    scores_string += display_stats(closeness_test, blinks_test, skip_n = SKIP_FIRST_FRAMES, test = True)
    scores_string += display_test_scores(closeness_test, closeness_predictions)
    # write_outputs(file_path, closeness_predictions_skipped, blink_predictions_skipped, \
    #           frame_info_df, scores_string)

    # # write test output files by using outputs of skip_first_n_frames() function
    # # no need to write frame_info_df and scores_string since they already have written above
    # write_outputs(file_path, closeness_test, blinks_test, test = True)
    # c_pred, b_pred, df, c_test, b_test, s_str= load_datasets("C:\\Users\\huynh14\\DMS\\scripts\\facelandmarks\\OpenSeeFace", "talking")

    # # check results
    # print(np.array(c_pred).shape, np.array(b_pred).shape)
    # print(np.array(c_test).shape, np.array(b_test).shape)
    # print()
    # print(s_str)

    
