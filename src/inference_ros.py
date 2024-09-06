# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import norm
from scipy.stats import circmean, circstd
import tensorflow.keras.backend as K

from deep_orientation import beyer    # noqa # pylint: disable=unused-import
from deep_orientation import mobilenet_v2    # noqa # pylint: disable=unused-import
from deep_orientation import beyer_mod_relu     # noqa # pylint: disable=unused-import

from deep_orientation.inputs import INPUT_TYPES
from deep_orientation.inputs import INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.outputs import (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION,
                                      OUTPUT_BITERNION)
import deep_orientation.preprocessing as pre
import deep_orientation.postprocessing as post

import utils.img as img_utils
from utils.io import get_files_by_extension

import time

import rospy
from std_srvs.srv import Trigger, TriggerResponse

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion
from tf.transformations import quaternion_from_euler
import math

import tensorflow as tf
# from keras import backend as K

# seaborn.set_style('darkgrid')
seaborn.set_context('notebook', font_scale=1.2)

HEIGHT = 720
WIDTH = 1280

HORIZONTAL_FOV = 70

def _parse_args():
    """Parse command-line arguments"""
    desc = 'Apply neural network for orientation estimation'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('model',
                        type=str,
                        help=("Model to use: beyer, beyer_mod_relu or "
                              "mobilenet_v2"),
                        choices=['beyer', 'beyer_mod_relu', 'mobilenet_v2'])

    parser.add_argument('weights_filepath',
                        type=str,
                        help="Path to the weights to load")

    parser.add_argument('image_or_image_basepath',
                        type=str,
                        help=("Path to a single image file or to a directory "
                              "containing multiple image files"))

    # input -------------------------------------------------------------------
    parser.add_argument('-it', '--input_type',
                        type=str,
                        default=INPUT_DEPTH,
                        choices=INPUT_TYPES,
                        help=(f"Input type. One of {INPUT_TYPES}, default: "
                              f"{INPUT_DEPTH}"))

    parser.add_argument('-iw', '--input_width',
                        type=int,
                        default=46,
                        help="Patch width to use, default: 96")

    parser.add_argument('-ih', '--input_height',
                        type=int,
                        default=46,
                        help="Patch height to use, default: 96")

    parser.add_argument('-ip', '--input_preprocessing',
                        type=str,
                        default='standardize',
                        choices=['standardize', 'scale01', 'none'],
                        help="Preprocessing to apply. One of [standardize, "
                             "scale01, none], default: standardize")

    # output ------------------------------------------------------------------
    parser.add_argument('-n', '--n_samples',
                        type=int,
                        default=1,
                        help="If `n_samples` > 1, dropout sampling is applied,"
                             " default: 1")

    parser.add_argument('-ot', '--output_type',
                        type=str,
                        default=OUTPUT_BITERNION,
                        choices=OUTPUT_TYPES,
                        help=(f"Output type. One of {OUTPUT_TYPES}, default: "
                              f"{OUTPUT_BITERNION})"))

    parser.add_argument('-nc', '--n_classes',
                        type=int,
                        default=8,
                        help=(f"Number of classes when output_type is "
                              f"{OUTPUT_CLASSIFICATION}, default: 8"))

    # other -------------------------------------------------------------------
    parser.add_argument('-ma', '--mobilenet_v2_alpha',
                        type=float,
                        choices=[0.35, 0.5, 0.75, 1.0],
                        default=1.0,
                        help="Alpha value for MobileNet v2 (default: 1.0)")

    parser.add_argument('-d', '--devices',
                        type=str,
                        default='0',
                        help="GPU device id(s) to use. (default: 0)")

    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        default=False,
                        help="CPU only, do not run with GPU support")

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="Enable verbose output")

    # return parsed args
    return parser.parse_args()


def load_network(model_name, weights_filepath,
                 input_type, input_height, input_width,
                 output_type,
                 sampling=False,
                 **kwargs):

    # load model --------------------------------------------------------------
    model_module = globals()[model_name]
    model_kwargs = {}
    if model_name == 'mobilenet_v2' and 'mobilenet_v2_alpha' in kwargs:
        model_kwargs['alpha'] = kwargs.get('mobilenet_v2_alpha')
    if output_type == OUTPUT_CLASSIFICATION:
        assert 'n_classes' in kwargs
        model_kwargs['n_classes'] = kwargs.get('n_classes')

    model = model_module.get_model(input_type=input_type,
                                   input_shape=(input_height, input_width),
                                   output_type=output_type,
                                   sampling=sampling,
                                   **model_kwargs)

    # load weights ------------------------------------------------------------
    model.load_weights(weights_filepath)

    return model

def getHumanPosition(image_number):
    bounding_boxes_dict = eval(self.bounding_boxes.data)
    
    x1, y1, x2, y2 = bounding_boxes_dict[image_number]
    
    
    # depth image, distance calculation
    try:
        depth_image = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough')
    except CvBridgeError as e:
        rospy.logerr(e)
        return
    
    bb_center_point_x, bb_center_point_y = (x1 + y1) / 2, (x2 + y2) / 2
    
    depth_height, depth_width = depth_image.shape

    x_center_depth = x_center * depth_width / WIDTH
    y_center_depth = y_center * depth_height / HEIGHT
    
    depth = 0
    
    # This method can be made better
    if 0 <= x_center_depth < depth_width and 0 <= y_center_depth < depth_height:
        depth = depth_image[int(floor(y_center_depth)), int(floor(x_center_depth))]
    else:
        rospy.logwarn("Bounding box center is out of depth image bounds.")
        
    # distance tells the distance of the object (here human) from the depth camera
    
    camera_c_x, camera_c_y = WIDTH / 2, HEIGHT / 2
    
    angle_per_pixel = HORIZONTAL_FOV / WIDTH
    
    horizontal_offset_from_center = (bb_center_point_x - camera_c_x)
    
    angle = horizontal_offset_from_center * angle_per_pixel
    
    # Now we have r, θ: depth, angle
    
    ############# For now, I am assuming that the robot is stationary. Once I figure out things with that, I would come back to fix it, ig.
    
    # Local X: front of robot
    # Local Y: Orthogonal of local x in the plane of movement of the robot
    # Local Z: Local X x Local Y
    
    angle_in_radians = math.radians(angle)
    
    x_coord = depth * math.cos(angle_in_radians)
    y_coord = depth * math.sin(angle_in_radians)
    
    # for now, I am keeping the height of the person to be a constant
    z_coord = 10
    
    return x_coord, y_coord, z_coord

model = None
args = None
filepaths = None
len_cnt = None


# def setup():
# global model, args, filepaths, len_cnt

args = _parse_args()

# set device and data format ----------------------------------------------
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.devices = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
if not args.devices or args.model == 'mobilenet_v2':
    # note: tensorflow supports b01c pooling on cpu only
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')

# load model --------------------------------------------------------------
# Create the graph and set as default
graph = tf.get_default_graph()

# Ensure that all operations are within this graph context
with graph.as_default():
    # load your model here
    model = load_network(args.model, args.weights_  filepath,
                        args.input_type, args.input_height, args.input_width,
                        args.output_type,
                        sampling=args.n_samples > 1,
                        n_classes=args.n_classes,
                        mobilenet_v2_alpha=args.mobilenet_v2_alpha)

marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

def main():
    # parse args --------------------------------------------------------------
    global model, args, filepaths, len_cnt, graph, marker_pub

    # parse for image files ---------------------------------------------------
    # note: we do not search for mask files, but derive masks from either the
    # depth or rgb image during preprocessing
    DEPTH_SUFFIX = '_Depth.pgm'
    RGB_SUFFIX = '_RGB.png'
    MASK_SUFFIX = '_Mask.png'
    # get filepaths
    mask_filepaths = get_files_by_extension(
            args.image_or_image_basepath, extension=MASK_SUFFIX.lower(),
            flat_structure=True, recursive=True, follow_links=True)

    if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
        depth_filepaths = get_files_by_extension(
            args.image_or_image_basepath, extension=DEPTH_SUFFIX.lower(),
            flat_structure=True, recursive=True, follow_links=True)
        assert len(depth_filepaths) == len(mask_filepaths)
        filepaths = list(zip(depth_filepaths, mask_filepaths))
        assert all(depth_fp.replace(DEPTH_SUFFIX, '') ==
                   mask_fp.replace(MASK_SUFFIX, '')
                   for depth_fp, mask_fp in filepaths)

    if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
        rgb_filepaths = get_files_by_extension(
            args.image_or_image_basepath, extension=RGB_SUFFIX.lower(),
            flat_structure=True, recursive=True, follow_links=True)
        assert len(rgb_filepaths) == len(mask_filepaths)
        filepaths = list(zip(rgb_filepaths, mask_filepaths))
        assert all(rgb_fp.replace(RGB_SUFFIX, '') ==
                   mask_fp.replace(MASK_SUFFIX, '')
                   for rgb_fp, mask_fp in filepaths)

    if args.input_type == INPUT_DEPTH_AND_RGB:
        filepaths = list(zip(depth_filepaths, rgb_filepaths, mask_filepaths))

    # define preprocessing function -------------------------------------------
    def load_and_preprocess(inputs):
        # unpack inputs
        if args.input_type == INPUT_DEPTH_AND_RGB:
            depth_filepath, rgb_filepath, mask_filepath = inputs
        elif args.input_type == INPUT_DEPTH:
            depth_filepath, mask_filepath = inputs
        else:
            rgb_filepath, mask_filepath = inputs

        # pack shape
        shape = (args.input_height, args.input_width)

        # load mask
        mask = img_utils.load(mask_filepath)
        mask_resized = pre.resize_mask(mask, shape)
        mask_resized = mask_resized > 0

        # prepare depth input
        if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
            # load
            depth = img_utils.load(depth_filepath)

            # create mask
            # mask = depth > 0
            # mask_resized = pre.resize_mask(mask.astype('uint8')*255, shape) > 0

            # mask (redundant, since mask is derived from depth image)
            # depth = pre.mask_img(depth, mask)

            # resize
            depth = pre.resize_depth_img(depth, shape)

            # 01 -> 01c
            depth = depth[..., None]

            # preprocess
            depth = pre.preprocess_img(
                depth,
                mask=mask_resized,
                scale01=args.input_preprocessing == 'scale01',
                standardize=args.input_preprocessing == 'standardize',
                zero_mean=True,
                unit_variance=True)

            # convert to correct data format
            if K.image_data_format() == 'channels_last':
                axes = 'b01c'
            else:
                axes = 'bc01'
            depth = img_utils.dimshuffle(depth, '01c', axes)

            # repeat if sampling is enabled
            if args.n_samples > 1:
                depth = np.repeat(depth, args.n_samples, axis=0)

        # prepare rgb input
        if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
            # load
            rgb = img_utils.load(rgb_filepath)

            # create mask
            # if args.input_type == INPUT_RGB:
            #     # derive mask from rgb image
            #     mask = rgb > 0
            #     mask_resized = pre.resize_mask(mask.astype('uint8')*255,
            #                                    shape) > 0
            # else:
            #     # mask rgb image using mask derived from depth image
            #    rgb = pre.mask_img(rgb, mask)

            # resize
            rgb = pre.resize_depth_img(rgb, shape)

            # preprocess
            rgb = pre.preprocess_img(
                rgb,
                mask=mask_resized,
                scale01=args.input_preprocessing == 'scale01',
                standardize=args.input_preprocessing == 'standardize',
                zero_mean=True,
                unit_variance=True)

            # convert to correct data format
            if K.image_data_format() == 'channels_last':
                axes = 'b01c'
            else:
                axes = 'bc01'
            rgb = img_utils.dimshuffle(rgb, '01c', axes)

            # repeat if sampling is enabled
            if args.n_samples > 1:
                rgb = np.repeat(rgb, args.n_samples, axis=0)

        # return preprocessed images
        if args.input_type == INPUT_DEPTH_AND_RGB:
            return depth, rgb
        elif args.input_type == INPUT_DEPTH:
            return depth,
        else:
            return rgb,

    # define postprocessing function ------------------------------------------
    def postprocess(output):
        if args.output_type == OUTPUT_BITERNION:
            return post.biternion2deg(output)
        elif args.output_type == OUTPUT_REGRESSION:
            return post.rad2deg(output)
        else:
            return post.class2deg(np.argmax(output, axis=-1), args.n_classes)

    # process files -----------------------------------------------------------
    len_cnt = len(str(len(filepaths)))
    # plt.ion()
    # fig = plt.figure(figsize=(8, 6))
    for i, inputs in enumerate(filepaths):
        print("[{:0{}d}/{:0{}d}]: {}".format(i+1, len_cnt, len(filepaths),
                                             len_cnt, inputs))
        # load and preprocess inputs
        nw_inputs = load_and_preprocess(inputs)

        # Start timing
        start_time = time.time()  # Line 1: Before the predict function

        # Predict
        with graph.as_default():
            nw_output = model.predict(nw_inputs, batch_size=args.n_samples)


        # End timing
        end_time = time.time()  # Line 2: After the predict function
        
        # Calculate the time taken for the prediction
        time_taken = end_time - start_time
        print(f"Time taken for prediction: {time_taken:.4f} seconds")

        # postprocess output
        output = postprocess(nw_output)
        
        print(output)

        # visualize inputs and predicted angle
        
        marker = Marker()
        marker.header.frame_id = "base_link" 
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "my_namespace"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set the position
        human_position_x, human_position_y, human_position_z = getHumanPosition(i)
        marker.pose.position = Point(human_position_x, human_position_y, human_position_z)

        # Convert orientation from degrees to radians and then to quaternion
        yaw_degrees = output[0]  # Change this to your desired angle in degrees
        yaw_radians = math.radians(yaw_degrees)
        quaternion = quaternion_from_euler(0.0, 0.0, yaw_radians)
        
        marker.pose.orientation = Quaternion(*quaternion)

        # Set the scale of the marker (1x1x1 here means the arrow has a length of 1m)
        marker.scale.x = 1.0  # Arrow length
        marker.scale.y = 0.1  # Arrow width
        marker.scale.z = 0.1  # Arrow height

        # Set the color (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Don't forget to set the alpha!

        # Publish the marker
        marker_pub.publish(marker)
       
def boundingboxes_callback(data):
    self.bounding_boxes = data

def rgb_image_callback(data):
    self.rgb_image = data

def depth_call_back(data):
    self.depth_image = data

def ros_node():
    """
    Main ROS node that sets up the service and handles incoming requests.
    """
    rospy.init_node('orientation_estimator')  # Initialize the ROS node
    
    self.full_rgb_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, rgb_image_callback)
    self.full_depth_image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_call_back)
    
    self.bounding_boxes_sub = rospy.Subscriber("/pose/bounding_boxes", String, boundingboxes_callback)
    
    rospy.Service('/pose/estimate', Trigger, handle_estimation_request)  # Create service

    rospy.loginfo("Orientation Estimator Node Initialized. Waiting for requests...")
    rospy.spin()  # Keep the node running

def handle_estimation_request(req):
    """
    Handle the incoming service request to process orientation estimation.
    """
    main()

    return TriggerResponse(
        success=True,
        message="Orientation estimation and continuation completed successfully."
    )

if __name__ == "__main__":
    
    # setup()
    
    # Run the ROS node
    ros_node()
