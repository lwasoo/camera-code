# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import asyncio

import cv2
import numpy as np
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
######################
#import numpy as np
import mediapipe as mp
#from mediapipe import python

from mediapipe.solutions import pose as mp_pose
from mediapipe.python.solutions.drawing_utils import draw_landmarks
#from mediapipe.tasks.python import vision
#from mediapipe.python.solutions import drawing_utils, pose
######################
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
######################

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

###############################

async def main(address: str, port: int, stream_every_n: int) -> None:
    # configure the camera client
    config = ClientConfig(address=address, port=port)
    client = OakCameraClient(config)

    # Get the streaming object from the service
    response_stream = client.stream_frames(every_n=stream_every_n)

    while True:
        # check the service state
        state = await client.get_state()
        if state.value != service_pb2.ServiceState.RUNNING:
            print("Camera is not streaming!")
            continue

        response: oak_pb2.StreamFramesReply = await response_stream.read()

        if response:
            # get the sync frame
            frame: oak_pb2.OakSyncFrame = response.frame
            print(f"Got frame: {frame.sequence_num}")
            print(f"Device info: {frame.device_info}")
            print(f"Timestamp: {frame.rgb.meta.timestamp}")
            print("#################################\n")
            
            try:
                # cast image data bytes to numpy and decode
                # NOTE: explore frame.[rgb, disparity, left, right]
                image = np.frombuffer(frame.rgb.image_data, dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                print("type:",type(image)) 
                ########################
                img = image
                # STEP 1: Import the necessary modules.
                

 #               mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

                # STEP 2: Create an ObjectDetector object.
                base_options = python.BaseOptions(model_asset_path='/data/home/amiga/camera/efficientnet_lite0_int8_2.tflite')
                options = vision.ObjectDetectorOptions(base_options=base_options,
                                                    score_threshold=0.5)
                detector = vision.ObjectDetector.create_from_options(options)

                # STEP 3: Load the input image.
                #****************
                #image = mp.Image.create_from_file(IMAGE_FILE)
                print(2)
                BGRimage=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                print("1")
                #***************
                # STEP 4: Detect objects in the input image.
                detection_result = detector.detect(BGRimage)

                # STEP 5: Process the detection result. In this case, visualize it.
                image_copy = np.copy(BGRimage.numpy_view())
                #image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img)

                # 获取关键点列表
                landmarks = results.pose_landmarks.landmark

                # 计算肩部高度和脚部高度
                shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * img.shape[0]
                foot_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * img.shape[0]
                #角度
                #print(shoulder_y)
                #print(foot_y)
                angle = abs(shoulder_y - foot_y) / (foot_y + 1e-6)
                if angle < 0.65:
                    res="inclined"
                else:
                    res="standing"

                categories = detection_result.detections[0].categories
                categories[0].category_name = res
                category_name = categories[0].score
                categories[0].score = 0
                #detection_result.detections[1].categories[0].score = 1
                #print(category_name)
                #print(len(detection_result.detections))


                annotated_image = visualize(image_copy, detection_result)
                rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


                mp.solutions.drawing_utils.draw_landmarks(rgb_annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)



                #cv2.imshow("img",rgb_annotated_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                print("type_out",type(rgb_annotated_image))
                print(rgb_annotated_image.shape)
                
                #print(image_jv.shape)
                # visualize the image
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.imshow("image", rgb_annotated_image)
                cv2.waitKey(1)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="amiga-camera-app")
    parser.add_argument("--port", type=int, required=True, help="The camera port.")
    parser.add_argument("--address", type=str, default="localhost", help="The camera address")
    parser.add_argument("--stream-every-n", type=int, default=1, help="Streaming frequency")
    args = parser.parse_args()

    asyncio.run(main(args.address, args.port, args.stream_every_n))

