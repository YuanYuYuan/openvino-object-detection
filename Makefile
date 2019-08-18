# ===== Object Detection Configs =====
FLAGS += --model './models/ssd-inception-v2-coco/ssd-inception-v2-coco.xml'
FLAGS += --weights './models/ssd-inception-v2-coco/ssd-inception-v2-coco.bin'
FLAGS += --labels './models/mscoco_label_map.yaml'
FLAGS += --threshold 0.5
FLAGS += --output './output.mp4'
FLAGS += --gui  # Toggle GUI


# ===== Video Streams Configs =====
RECORD_FILE = "./videos/record.mkv"
VIDEO_SOURCE = "http://viewer:viewer@117.56.55.194/Media/Streaming?type=jpeg&deviceid=20"  # 九份老街入口
# VIDEO_SOURCE = "http://60.250.155.175/t14a-d61a0c91"  # 武嶺停車場
# VIDEO_SOURCE = "https://www.eocemis.gov.taipei/cctv/d01c07e1"  # 台北信義區 市民大道-松山路口
# VIDEO_SOURCE = "http://cctvn04.freeway.gov.tw:8080/stream/GetStreamVideo?pm=163,A43,16"  # 國道1號(高公局交流道到五股交流道)


# ===== Sample Configs =====
SAMPLE_VIDEO = './videos/bus_station.mp4'
# SAMPLE_VIDEO = './videos/motorcycle.mp4'
# SAMPLE_VIDEO = './videos/scooters.mp4'


# ===== Input Configs =====
INPUT_FLAGS += --input-type 'file'  # video file
INPUT_FLAGS += --input 'YOUR_CUSTOM_VIDEO'
# INPUT_FLAGS += --input $(RECORD_FILE)

# INPUT_FLAGS += --input-type 'camera' # usb camera
# INPUT_FLAGS += --input 0


# ===== Live Input Configs =====
LIVE_INPUT_FLAGS += --input-type 'camera'  # live stream
LIVE_INPUT_FLAGS += --input '1'


# ===== Recipe =====
download_sample_videos:
	@ cd ./videos && make download

demo: download_sample_videos
	@ ./src/main.py $(FLAGS) --input-type 'file'  --input $(SAMPLE_VIDEO)

detect:
	@ ./src/main.py $(FLAGS) $(INPUT_FLAGS)

record:
	@ echo Recording video into $(RECORD_FILE), press CTRL+C to terminate.
	@ gst-launch-1.0 -v \
		souphttpsrc location=$(VIDEO_SOURCE)\
		do-timestamp=true \
		! multipartdemux \
		! image/jpeg,width=640,height=360 \
		! matroskamux \
		! queue \
		! filesink location=$(RECORD_FILE) \
		> /dev/null 2>&1
play:
	@ mpv $(RECORD_FILE)

stream:
	@ gst-launch-1.0 -v \
		souphttpsrc location=$(VIDEO_SOURCE)\
		! decodebin \
		! videoconvert \
		! videoscale \
		! queue \
		! ximagesink

live_detect:
	@ ./src/stream_loopback.sh $(VIDEO_SOURCE) &
	@ ./src/main.py $(FLAGS) $(LIVE_INPUT_FLAGS)
	@ pkill gst-launch-1.0
