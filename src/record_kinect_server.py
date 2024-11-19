#!/usr/bin/env python3

"""
ROS service to record audio and capture images from an Azure Kinect device. They are saved as WAV and JPG files

Service Details:
- Service Name: `record_kinect`
- Service Type: `audio_visual_localization/RecordKinect`
- Service Request Parameters:
  - `kinect_index` (int): The index of the Kinect device for audio recording.
  - `recording_duration` (float): Duration of audio recording in seconds.
  - `frames_dir` (str): Directory to save captured images.
  - `audio_dir` (str): Directory to save recorded audio files.
  - `timestamp` (str): Timestamp used in filenames for the recorded files.


Usage:
1. Launch kinect driver and run this node by using the command
$ roslaunch audio_visual_localization kinect_driver_server.launch

2. Use a ROS client to call the `record_kinect` service, such as the node neural_network_node.py
"""


import os
import rospy
from audio_visual_localization.srv import RecordKinect, RecordKinectResponse, RecordKinectRequest
import pyaudio
import wave
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess


def check_node_status(node_name):
    """
    Check the status of a specific ROS node.
    """
    try:
        output = subprocess.check_output(['rosnode', 'list']).decode('utf-8')
        return node_name in output
    except subprocess.CalledProcessError as e:
        rospy.logerr(f"Failed to query ROS nodes: {e}")
        return False


def record_audio_from_kinect(filename, kinect_index, recording_duration):
    """
    Record audio from the Azure Kinect microphone array and save it to a WAV file.
    """
    try:
        FORMAT = pyaudio.paInt16
        CHANNELS = 7    # Kinect Azure has 7 channels (mic array)
        RATE = 48000    # Sample rate for Kinect Azure
        CHUNK = 8192    # 1024  to support all the channels overload

        p = pyaudio.PyAudio()

        # Create a new audio stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=kinect_index,
                        frames_per_buffer=CHUNK)

        print("* Recording audio...")

        frames = []

        # Record in chunks
        for _ in range(int(recording_duration * RATE / CHUNK)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* Finished recording")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the audio data to a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"Audio saved to {filename}")
        return True

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return False

    except IOError as e:
        print("IOError: {e}")
        return False

    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

    finally:
        if 'stream' in locals():
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        if 'p' in locals():
            try:
                p.terminate()
            except Exception as e:
                print(f"Error terminating pyaudio: {e}")


def record_image_from_kinect(filename):
    """
    Capture an image from the ROS topic `/rgb/image_raw` and save it as an image file.
    """
    try:
        bridge = CvBridge()
        image_msg = rospy.wait_for_message('/rgb/image_raw', Image)
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv2.imwrite(filename, cv_image)
        rospy.loginfo(f"Saved image to {filename}")
        return True

    except Exception as e:
        rospy.logerr(f"Error capturing image: {e}")
        return False


def handle_record(req):
    """
    Handle the service request to record audio and capture an image.
    """
    if req.kinect_index is None:
        print("Azure Kinect Microphone Array not found.")
        return RecordKinectResponse(success=False)

    if not os.path.exists(req.frames_dir):
        os.makedirs(req.frames_dir)
    if not os.path.exists(req.audio_dir):
        os.makedirs(req.audio_dir)

    image_filename = os.path.join(req.frames_dir, f"{req.timestamp}.jpg")
    audio_filename = os.path.join(req.audio_dir, f"{req.timestamp}.wav")

    image_success = record_image_from_kinect(image_filename)
    audio_success = record_audio_from_kinect(audio_filename, req.kinect_index, req.recording_duration)

    return RecordKinectResponse(success=(audio_success and image_success))


def record_kinect_server():
    """
    Initializes the ROS node and advertises the `record_kinect_audio` service.
    """
    rospy.init_node('record_kinect_server')
    rospy.Service('record_kinect', RecordKinect, handle_record)
    rospy.loginfo("Ready to record and capture.")
    rospy.spin()


if __name__ == "__main__":
    try:
        record_kinect_server()
    except rospy.ROSInterruptException:
        pass
