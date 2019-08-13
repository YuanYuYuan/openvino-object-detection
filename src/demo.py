#!/usr/bin/env python3

import argparse
import cv2
import logging
import sys
from openvino.inference_engine import IENetwork, IEPlugin
from utils import label_map_util
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    default='../models/ssd-inception-v2-coco/ssd-inception-v2-coco.xml',
    help='trained model architecture (.xml)'
)
parser.add_argument(
    '--weights',
    default='../models/ssd-inception-v2-coco/ssd-inception-v2-coco.bin',
    help='trained model weights (.bin)'
)
parser.add_argument(
    '--input-type',
    default='file',
    choices=['file', 'camera'],
    help='video from file or camera'
)
parser.add_argument(
    '--input',
    default='../videos/demo.mp4',
    help='video input'
)
parser.add_argument(
    '--labels',
    default='../models/mscoco_label_map.pbtxt',
    help='labels mapping file'
)
parser.add_argument(
    '--threshold',
    default=0.5,
    type=float,
    help='probability threshold of predictions'
)
parser.add_argument(
    '--output',
    default='./output.mp4',
    help='save prediction into mp4 file'
)
parser.add_argument(
    '--gui',
    default=False,
    type=bool,
    help='toggle GUI'
)
args = parser.parse_args()

# Setup video source
if args.input_type == 'file':
    video_capture = cv2.VideoCapture(args.input)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
elif args.input_type == 'camera':
    video_capture = cv2.VideoCapture(0)

# Prepare labels map
labels_map = label_map_util.create_category_index_from_labelmap(
    args.labels,
    use_display_name=True
)

# setup logger
logging.basicConfig(
    format='[ %(levelname)s ] %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)

# Load model into inference engine
plugin = IEPlugin('MYRIAD')
net = IENetwork(
    model=args.model,
    weights=args.weights,
)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = plugin.load(
    network=net,
    num_requests=2,
)

# Input shape: [n_samples, n_channels, height, width]
input_shape = net.inputs[input_blob].shape
input_height = input_shape[2]
input_width = input_shape[3]
assert input_shape[0] == 1


def preprocess(frame):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32)
    frame = np.moveaxis(frame, -1, 0)  # change layout from HWC to CHW
    batch = np.expand_dims(frame, 0)  # convert into batch data with size 1
    return batch


cur_request_id = 0
next_request_id = 1

logging.info("Starting inference in async mode...")
logging.info("To switch between sync and async modes press Tab button")
logging.info("To stop the demo execution press Esc button")
is_async_mode = True
render_time = 0
ret, frame = video_capture.read()

video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

if args.output is not None:
    video_writer = cv2.VideoWriter(
        # 'outpy.avi',
        args.output,
        # fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=10,
        frameSize=(video_width, video_height)
    )


print("To close the application, press 'CTRL+C' or any key with focus on the output window")
while video_capture.isOpened():
    if is_async_mode:
        retrieved, next_frame = video_capture.read()
    else:
        retrieved, frame = video_capture.read()
    if not retrieved:
        break

    # Main sync point:
    # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
    # in the regular mode we start the CURRENT request and immediately wait for it's completion
    inf_start = time.time()
    exec_net.start_async(
        request_id=next_request_id if is_async_mode else cur_request_id,
        inputs={input_blob: preprocess(next_frame if is_async_mode else frame)}
    )
    if exec_net.requests[cur_request_id].wait(-1) == 0:
        inf_end = time.time()
        det_time = inf_end - inf_start

        # Parse detection results of the current request
        result = exec_net.requests[cur_request_id].outputs[out_blob]
        for obj in result[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > args.threshold:
                x_min = int(obj[3] * video_width)
                y_min = int(obj[4] * video_height)
                x_max = int(obj[5] * video_width)
                y_max = int(obj[6] * video_height)
                class_id = int(obj[1])
                # Draw box and label\class_id
                color = (
                    min(class_id * 12.5, 255),
                    min(class_id * 7, 255),
                    min(class_id * 5, 255)
                )
                cv2.rectangle(
                    frame,
                    pt1=(x_min, y_min),
                    pt2=(x_max, y_max),
                    color=color,
                    thickness=2
                )
                label = labels_map[class_id]['name']
                probability = round(obj[2] * 100, 1)
                cv2.putText(
                    frame,
                    text='%s, %.2f %%' % (label, probability),
                    org=(x_min, y_min - 7),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6,
                    color=color,
                    thickness=1
                )

        # Draw performance stats
        inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
            "Inference time: {:.3f} ms".format(det_time * 1000)
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
        async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
            "Async mode is off. Processing request {}".format(cur_request_id)

        cv2.putText(
            frame,
            inf_time_message,
            (15, 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (200, 10, 10),
            1
        )
        cv2.putText(
            frame,
            render_time_message,
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (10, 10, 200),
            1
        )
        cv2.putText(
            frame,
            async_mode_message,
            (10, int(video_height - 20)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (10, 10, 200),
            1
        )

    render_start = time.time()
    if args.gui:
        cv2.imshow("Detection Results", frame)
    if args.output is not None:
        video_writer.write(frame)
    render_end = time.time()
    render_time = render_end - render_start

    if is_async_mode:
        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame

    key = cv2.waitKey(1)
    if key == 27:
        break
    if (9 == key):
        is_async_mode = not is_async_mode
        logging.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
        time.sleep(0.1)

video_capture.release()
if args.output is not None:
    video_writer.release()
cv2.destroyAllWindows()
