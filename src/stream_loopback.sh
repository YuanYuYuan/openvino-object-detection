#!/usr/bin/env bash

VIDEO_SOURCE="$1"
if [ -z $VIDEO_SOURCE ]; then
    echo "Requires stream url as argument."
    exit
fi

DEVICE_ID="1"
DEVICE="/dev/video$DEVICE_ID"
if [ ! -e $DEVICE ]; then
    if $(lsmod | grep -q v4l2loopback); then
        sudo modprobe -r v4l2loopback
    fi
    sudo modprobe v4l2loopback video_nr=$DEVICE_ID
    echo "Created the loopback device $DEVICE"
fi

pkill gst-launch-1.0
echo "Streamiing..."
gst-launch-1.0 -v \
    souphttpsrc location=$VIDEO_SOURCE\
    ! image/jpeg,width=640,height=360 \
    ! decodebin \
    ! videoconvert \
    ! videoscale \
    ! queue \
    ! tee \
    ! v4l2sink device=$DEVICE > /dev/null 2>&1
