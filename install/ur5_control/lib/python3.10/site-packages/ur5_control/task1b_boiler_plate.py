#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Logistic coBot (LB) Theme (eYRC 2024-25)
*        		===============================================
*
*  This script should be used to implement Task 1B of Logistic coBot (LB) Theme (eYRC 2024-25).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1b_boiler_plate.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
import cv2.aruco as aruco


##################### FUNCTION DEFINITIONS #######################

def calculate_rectangle_area(coordinates):
    '''
    Description:    Function to calculate area or detected aruco

    Args:
        coordinates (list):     coordinates of detected aruco (4 set of (x,y) coordinates)

    Returns:
        area        (float):    area of detected aruco
        width       (float):    width of detected aruco
    '''

    ############ Function VARIABLES ############

    # You can remove these variables after reading the instructions. These are just for sample.

    ############ ADD YOUR CODE HERE ############

    # INSTRUCTIONS & HELP : 
    #	->  Recevice coordiantes from 'detectMarkers' using cv2.aruco library 
    #       and use these coordinates to calculate area and width of aruco detected.
    #	->  Extract values from input set of 4 (x,y) coordinates 
    #       and formulate width and height of aruco detected to return 'area' and 'width'.
    for i, coordinate in enumerate(coordinates):
        # Extract the coordinates of the marker
        pts = coordinate[0]  # Get the corner points

        # Calculate the width and height of the marker
        width = np.linalg.norm(pts[0] - pts[1])
        height = np.linalg.norm(pts[1] - pts[2])
        
        # Calculate area
        area = width * height

        # print(f'Width: {width:.2f}, Height: {height:.2f}, Area: {area:.2f}')

    ############################################

    return area, width

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # print(x,y,z)
    return np.array([x, y, z])


def detect_aruco(image):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''

    ############ Function VARIABLES ############

    # ->  You can remove these variables if needed. These are just for suggestions to let you get started

    # Use this variable as a threshold value to detect aruco markers of certain size.
    # Ex: avoid markers/boxes placed far away from arm's reach position  
    aruco_area_threshold = 1500

    # The camera matrix is defined as per camera info loaded from the plugin used. 
    # You may get this from /camer_info topic when camera is spawned in gazebo.
    # Make sure you verify this matrix once if there are calibration issues.
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])

    # The distortion matrix is currently set to 0. 
    # We will be using it during Stage 2 hardware as Intel Realsense Camera provides these camera info.
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])

    # We are using 150x150 aruco marker size
    size_of_aruco_m = 0.15

    # You can remove these variables after reading the instructions. These are just for sample.
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_list = []
    detected_ids = []
    ids = []
    
 
    ############ ADD YOUR CODE HERE ############

    # INSTRUCTIONS & HELP : 
    imgGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #	->  Convert input BGR image to GRAYSCALE for aruco detection
    arucoDict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParam=aruco.DetectorParameters()

    corners,detected_ids,rejected= aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)
     
    # print(ids)
    # print(bboxs)

    #   ->  Use these aruco parameters-
    #       ->  Dictionary: 4x4_50 (4x4 only until 50 aruco IDs)

    #   ->  Detect aruco marker in the image and store 'corners' and 'ids'
    #       ->  HINT: Handle cases for empty markers detection. 

    #   ->  Draw detected marker on the image frame which will be shown later

    #   ->  Loop over each marker ID detected in frame and calculate area using function defined above (calculate_rectangle_area(coordinates))

    #   ->  Remove tags which are far away from arm's reach positon based on some threshold defined

    #   ->  Calculate center points aruco list using math and distance from RGB camera using pose estimation of aruco marker
    #       ->  HINT: You may use numpy for center points and 'estimatePoseSingleMarkers' from cv2 aruco library for pose estimation

    # if len(detected_ids) > 0:
    #     for i, corner in enumerate(corners):
    #         area,width=calculate_rectangle_area(corners)
            
    #         if area >= aruco_area_threshold:
    #             ids.append(detected_ids[i][0])

    # if len(detected_ids) > 0:
    for i, corner in enumerate(corners):

            area,width=calculate_rectangle_area(corners)

            if area >= aruco_area_threshold:
            
                # Draw a square around the markers
                aruco.drawDetectedMarkers(image,corners,detected_ids)
                # area,width=calculate_rectangle_area(corners)

                width_list.append(width)

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.01, cam_mat, dist_mat)

                # print(rvec)

                cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 0.01)

                pts = corner[0]
                center_points = np.mean(pts, axis=0)
                center_aruco_list.append(center_points)
                
                # cv2.circle(image, (int(center_points[0]), int(center_points[1])), 10, (0, 255, 255), -1)

                distance = np.linalg.norm(tvec)  # Depth (z)
                distance_from_rgb_list.append(distance)

                # Draw Axis
                # rotation_matrix, _ = cv2.Rodrigues(rvec) #rotational_matrix

                # # Extract angles (in radians) from the rotation matrix
                # pitch = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])  # Pitch
                # roll = np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2))  # Roll
                # yaw = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])  # Yaw

                # if abs(R[2, 0]) != 1:
                #     pitch = -np.arcsin(R[2, 0])
                #     roll = np.arctan2(R[2, 1], R[2, 2])
                #     yaw = np.arctan2(R[1, 0], R[0, 0])
                # else:
                # # Gimbal lock case (when R[2,0] == ±1)
                #     yaw = 0  # or any other value, gimbal lock prevents unique solution
                #     if R[2, 0] == -1:
                #         pitch = np.pi / 2
                #         roll = np.arctan2(R[0, 1], R[0, 2])
                #     else:
                #         pitch = -np.pi / 2
                #         roll = np.arctan2(-R[0, 1], -R[0, 2])


                # # Store angles in degrees
                # angles = (roll, pitch, yaw)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                euler_angles=rotationMatrixToEulerAngles(rotation_matrix)
                angle_aruco_list.append(np.degrees(euler_angles))
                

                ids.append(detected_ids[i][0])
                
        
            # print(f'Angles (Yaw, Pitch, Roll) for Marker ID {detected_ids[i]}: {angles}')

            # width = np.linalg.norm(pts[0] - pts[1])  # Example for width based on corner points

            # Assuming center_aruco_list is defined and populated

        # print("Ids:")
        # for id in ids:
        #     print(f"id: {id}")
        # print(angle_aruco_list)

        # for angle_aruco in angle_aruco_list:
        # # Correct angle using provided formula
        #     # angle_aruco = angle_aruco_list[i]
        #     print(f"id: {angle_aruco}")

    # for dist in distance_from_rgb_list:
    #     # Correct angle using provided formula
    #         # angle_aruco = angle_aruco_list[i]
    #     print(f"id: {dist}")

    #   ->  Draw frame axes from coordinates received using pose estimation
    #       ->  HINT: You may use 'cv2.drawFrameAxes'

    ############################################

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_list, detected_ids, ids





##################### CLASS DEFINITION #######################

class aruco_tf(Node):
    '''
    ___CLASS___

    Description:    Class which servers purpose to define process for detecting aruco marker and publishing tf on pose estimated.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class aruco_tf
                        All classes have a function called __init__(), which is always executed when the class is being initiated.
                        The __init__() function is called automatically every time the class is being used to create a new object.
                        You can find more on this topic here -> https://www.w3schools.com/python/python_classes.asp
        '''

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        self.cv_image = None                                                        # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                       # depth image variable (from depthimagecb())


    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic. 
                        Use this function to receive image depth data and convert to CV2 image

        Args:
            data (Image):    Input depth image frame received from aligned depth camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 

        #	->  Use data variable to convert ROS Image message to CV2 Image type

        #   ->  HINT: You may use CvBridge to do the same

        depth_image1 = self.bridge.imgmsg_to_cv2(data, desired_encoding='16UC1') # desired_encoding = 'passthrough'

        # Process the depth image (for example, normalize it for display)
        normalized_depth = cv2.normalize(depth_image1, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = np.uint8(normalized_depth)  # Convert to 8-bit image for display
        self.depth_image=depth_image1
        
        # Optionally, display the depth image
        # cv2.imshow("Depth Image", normalized_depth)
        # # cv2.waitKey(1)

        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        #     cv2.destroyAllWindows()

        #return depth_image

        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.
                        Use this function to receive raw image data and convert to CV2 image

        Args:
            data (Image):    Input coloured raw image frame received from image_raw camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 

        #	->  Use data variable to convert ROS Image message to CV2 Image type

        #   ->  HINT:   You may use CvBridge to do the same
        #               Check if you need any rotation or flipping image as input data maybe different than what you expect to be.
        #               You may use cv2 functions such as 'flip' and 'rotate' to do the same

        ############################################
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.cv_image=frame
        #detect_aruco(self.cv_image)
        # cv2.imshow("output",self.cv_image)
        cv2.waitKey(1)

        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        #     cv2.destroyAllWindows()

        #return frame

    def publish_tf(self, marker_id, position, orientation):
        # Create a TransformStamped message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()  # Use ROS 2 clock for the timestamp
        t.header.frame_id = "camera_depth_frame"
        t.child_frame_id = f"cam_{marker_id}"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = position
        t.transform.rotation.x=orientation[0]
        t.transform.rotation.y=orientation[1]
        t.transform.rotation.z=orientation[2]
        t.transform.rotation.w=orientation[3]

        # Publish the transform
        self.br.sendTransform(t)

        # listener = self.listener()
        # transform = self.tf_buffer.lookup_transform("base_link", "camera_depth_frame", self.get_clock().now().to_msg())

        # obj_transform = TransformStamped()
        # obj_transform.header.stamp = self.get_clock().now().to_msg()
        # obj_transform.header.frame_id = "base_link"
        # obj_transform.child_frame_id = f"obj_{marker_id}"
        # obj_transform.transform.translation.x = transform.transform.translation.x
        # obj_transform.transform.translation.y = transform.transform.translation.y
        # obj_transform.transform.translation.z = transform.transform.translation.z
        # obj_transform.transform.rotation.x = transform.transform.rotation.x
        # obj_transform.transform.rotation.y = transform.transform.rotation.y
        # obj_transform.transform.rotation.z = transform.transform.rotation.z
        # obj_transform.transform.rotation.w = transform.transform.rotation.w

        # self.br.sendTransform(obj_transform)

        from_frame_rel=f"cam_{marker_id}"                                                                     
        to_frame_rel = "base_link"                                                                     # frame to which transfrom has been sent

        try:
            transform = self.tf_buffer.lookup_transform( to_frame_rel, from_frame_rel, rclpy.time.Time())       # look up for the transformation between 'obj_1' and 'base_link' frames
            obj_transform = TransformStamped()
            obj_transform.header.stamp = self.get_clock().now().to_msg()
            obj_transform.header.frame_id = "base_link"
            obj_transform.child_frame_id = f"obj_{marker_id}"
            obj_transform.transform.translation.x = transform.transform.translation.x 
            obj_transform.transform.translation.y = transform.transform.translation.y 
            obj_transform.transform.translation.z = transform.transform.translation.z 
            obj_transform.transform.rotation.x = transform.transform.rotation.x
            obj_transform.transform.rotation.y = transform.transform.rotation.y
            obj_transform.transform.rotation.z = transform.transform.rotation.z
            obj_transform.transform.rotation.w = transform.transform.rotation.w

            # obj_transform.transform.translation = transform.transform.translation
            # obj_transform.transform.rotation = transform.transform.rotation


            self.br.sendTransform(obj_transform)

            self.get_logger().info(f'Successfully received data!')

        except tf2_ros.TransformException as e:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {e}')
            return

        # Logging transform data...
        # self.get_logger().info(f'Translation X:  {t.transform.translation.x}')
        # self.get_logger().info(f'Translation Y:  {t.transform.translation.y}')
        # self.get_logger().info(f'Translation Z:  {t.transform.translation.z}')
        # self.get_logger().info(f'Rotation X:  {t.transform.rotation.x}')                                # NOTE: rotations are in quaternions
        # self.get_logger().info(f'Rotation Y:  {t.transform.rotation.y}')
        # self.get_logger().info(f'Rotation Z:  {t.transform.rotation.z}')
        # self.get_logger().info(f'Rotation W:  {t.transform.rotation.w}')

    # def lookup_transform(self, target_frame, source_frame):
    #     transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
    #     return transform
            
 

    def process_image(self):
        '''
        Description:    Timer function used to detect aruco markers and publish tf on estimated poses.

        Args:
        Returns:
        '''
        

        ############ Function VARIABLES ############

        # These are the variables defined from camera info topic such as image pixel size, focalX, focalY, etc.
        # Make sure you verify these variable values once. As it may affect your result.
        # You can find more on these variables here -> http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        
        sizeCamX = 1280
        sizeCamY = 720 
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375
            

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 
        # self.colorimagecb(data)
        #	->  Get aruco center, distance from rgb, angle, width and ids list from 'detect_aruco_center' defined above
        image = self.cv_image
        depth_image = self.depth_image
        center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_list,detected_ids,ids = detect_aruco(image)

        #   ->  Loop over detected box ids received to calculate position and orientation transform to publish TF 
        for i, marker_id in enumerate(ids):
        # Correct angle using provided formula
            
            angle_aruco = float(angle_aruco_list[i][2])
           
            corrected_angle = (0.788 * angle_aruco) - ((angle_aruco ** 2.0) / 3160.0)
            roll = angle_aruco_list[i][0]
            pitch = angle_aruco_list[i][1]
            

            # print(roll,pitch,corrected_angle)

        # Calculate quaternion from corrected angle
            yaw = corrected_angle
    
            # orientation = tf2_ros.transformations.quaternion_from_euler(0.0, 0.0, yaw)
            r = R.from_euler('zxy', [yaw,pitch,roll], degrees=True)
            # orientation = r.as_quat()
            # q2=[0.0,-0.13013,0.0,0.991497]
            flip_z_inward = R.from_euler('y', 240, degrees=True) #240
            corrected_orientation = flip_z_inward * r
            orientation_flipped = corrected_orientation.as_quat()

            # q2=[0.0,-0.13013,0.0,0.991497]
                    
            # x1, y1, z1,w1 = q1

            # x2, y2, z2,w2 = q2
            
            # w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            # x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            # y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            # z = w1*z2 + x1*y2 - y1*x2 + z1*w2

            # orientation_flipped=[x,y,z,w]

            # q2=[np.sin(np.pi/2),0.0,0.0,1.0]
            
            # x1, y1, z1,w1 = orientation
            # x2, y2, z2,w2 = q2
            
            # w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            # x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            # y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            # z = w1*z2 + x1*y2 - y1*x2 + z1*w2

            # orientation=[x,y,z,w]
            # print(orientation[0])
            # norm_quat = np.linalg.norm(quaternion)
            # orientation = angle_aruco_list[i]

            # Retrieve distance from RGB camera
            distance_from_rgb = distance_from_rgb_list[i]/1000.0 # Convert mm to m
            cX, cY = center_aruco_list[i]
            cX = int(np.round(cX))
            cY = int(np.round(cY))
            # print(cX,cY)
            
           
            # Rectify x, y, z
            #z = distance_from_rgb
            # x = z * (sizeCamX - cX - centerCamX) / focalX 
            # y = z * (sizeCamY - cY - centerCamY) / focalY 
            # z = distance_from_rgb
            #depth_image= self.depth_image
            height, width = depth_image.shape
            # print(height,width)
            if 0 <= cX < width and 0 <= cY < height:
              depth_value_mm = depth_image[cY, cX]
              depth_value_m = depth_value_mm / 1000.0  # Convert to meters
            else:
                print(f"Error: cX ({cX}) or cY ({cY}) is out of bounds.")
                continue

            # image = self.cv_image
            # camera_matrix = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
            # def get_aruco_position_with_depth(image, depth_image, camera_matrix):
            #     # Detect the ArUco marker
            #     aruco_dict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            #     aruco_params=aruco.DetectorParameters()

            #     # Convert to grayscale if needed
            #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            #     # Detect markers
            #     corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=aruco_params)

            #     X,Y,Z = 0.0,0.0,0.0
            #     print(corners)
                

            #     if len(corners)>0:
            #         # Assuming we detect only one marker, refine the selection process if needed
            #         marker_corners = corners[0][0]  # Get corners of the first marker
            #         center_u = int(np.mean(marker_corners[:, 0]))  # X coordinate of marker center
            #         center_v = int(np.mean(marker_corners[:, 1]))  # Y coordinate of marker center
            #         print(center_u,center_v)
            #         # Get depth value at (center_u, center_v)
            #         depth_value_mm = depth_image[center_v, center_u]  # Depth at the marker center in millimeters
            #         depth_value_m = depth_value_mm / 1000.0  # Convert to meters
                    
            #         # Camera intrinsics
            #         fx = camera_matrix[0, 0]
            #         fy = camera_matrix[1, 1]
            #         cx = camera_matrix[0, 2]
            #         cy = camera_matrix[1, 2]

            #         # Compute real-world coordinates
            #         X = (center_u - cx) * depth_value_m / fx
            #         Y = (center_v - cy) * depth_value_m / fy
            #         Z = depth_value_m
            #     return X,Y,Z

            # x,y,z = get_aruco_position_with_depth(image, depth_image, camera_matrix)
            #print(x,y,z)
            
            

            # def get_3d_coordinates(cX, cY, distance_from_rgb, focalX, focalY, centerCamX, centerCamY,sizeCamX,sizeCamY):
            #     # Z is the depth (distance from the camera to the marker)
        
            
            # Calculate X and Y in 3D space
            x = (sizeCamX-cX - centerCamX) * depth_value_m / focalX
            y = (sizeCamY-cY - centerCamY) * depth_value_m / focalY
            z = depth_value_m 
            #     return x, y, z

            # x, y, z = get_3d_coordinates(cX, cY, distance_from_rgb, focalX, focalY, centerCamX, centerCamY, sizeCamX,sizeCamY)
            # # print(x,y,z)

            # Mark center points on the image frames
            cv2.circle(self.cv_image, (int(cX), int(cY)), 5, (255, 255, 0), -1)

            # Publish transform from camera_link to ArUco marker
            self.publish_tf(marker_id, (float(z),float(x),float(y)), orientation_flipped)
            # self.lookup_transform("base_link",)
            
            # cv2.imshow("Detected ArUco Markers", self.cv_image)
            # cv2.waitKey(1)
    

            # Lookup transform for base_link and obj frame
            # (Assuming lookup_transform is defined)


        # Show image with detected markers and centers
        cv2.imshow("Detected ArUco Markers", self.cv_image)
        cv2.waitKey(1)


        #   ->  Use this equation to correct the input aruco angle received from cv2 aruco function 'estimatePoseSingleMarkers' here
        #       It's a correction formula- 
        #       angle_aruco = (0.788*angle_aruco) - ((angle_aruco**2)/3160)

        #   ->  Then calculate quaternions from roll pitch yaw (where, roll and pitch are 0 while yaw is corrected aruco_angle)

        #   ->  Use center_aruco_list to get realsense depth and log them down. (divide by 1000 to convert mm to m)

        #   ->  Use this formula to rectify x, y, z based on focal length, center value and size of image
        #       x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
        #       y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
        #       z = distance_from_rgb
        #       where, 
        #               cX, and cY from 'center_aruco_list'
        #               distance_from_rgb is depth of object calculated in previous step
        #               sizeCamX, sizeCamY, centerCamX, centerCamY, focalX and focalY are defined above

        #   ->  Now, mark the center points on image frame using cX and cY variables with help of 'cv2.cirle' function 

        #   ->  Here, till now you receive coordinates from camera_link to aruco marker center position. 
        #       So, publish this transform w.r.t. camera_link using Geometry Message - TransformStamped 
        #       so that we will collect it's position w.r.t base_link in next step.
        #       Use the following frame_id-
        #           frame_id = 'camera_link'
        #           child_frame_id = 'cam_<marker_id>'          Ex: cam_20, where 20 is aruco marker ID

        #   ->  Then finally lookup transform between base_link and obj frame to publish the TF
        #       You may use 'lookup_transform' function to pose of obj frame w.r.t base_link 

        #   ->  And now publish TF between object frame and base_link
        #       Use the following frame_id-
        #           frame_id = 'base_link'
        #           child_frame_id = 'obj_<marker_id>'          Ex: obj_20, where 20 is aruco marker ID

        #   ->  At last show cv2 image window having detected markers drawn and center points located using 'cv2.imshow' function.
        #       Refer MD book on portal for sample image -> https://portal.e-yantra.org/

        #   ->  NOTE:   The Z axis of TF should be pointing inside the box (Purpose of this will be known in task 1C)
        #               Also, auto eval script will be judging angular difference as well. So, make sure that Z axis is inside the box (Refer sample images on Portal - MD book)

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
                    You can find more on this here -> https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/
    '''

    main()