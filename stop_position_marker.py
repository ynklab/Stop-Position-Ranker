import numpy as np
import scipy.ndimage as ndimage
import cv2
import os
import numpy as np
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scene import Scene


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def visualize_depth(disp_resized_np):
    print(np.size(disp_resized_np))
    vmax = np.percentile(disp_resized_np, 95)

    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma_r')
    # mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    name_dest_im = "visualize_depth.jpeg"
    # plt.imsave(name_dest_im, disp_resized_np, cmap='gray') # for saving as gray depth maps
    im.save(name_dest_im) # for saving as colored depth maps

def rotate_point(param, start_point, angle):
    x, y = param
    x0, y0 = start_point
    x_rot = x0 + np.cos(angle) * (x - x0) - np.sin(angle) * (y - y0)
    y_rot = y0 + np.sin(angle) * (x - x0) + np.cos(angle) * (y - y0)
    return int(x_rot), int(y_rot)

def check_remaining_space(edge_start, edge_end, box_corners, car_length_depth):
    """
    Check remaining space on an edge after placing a box

    Parameters:
    edge_start: tuple of (x,y) for edge start
    edge_end: tuple of (x,y) for edge end
    box_corners: list of (x,y) tuples representing the placed box corners
    car_length_depth: minimum length needed for a new car
    """
    # Calculate total edge length
    edge_length = np.sqrt((edge_end[0] - edge_start[0]) ** 2 + (edge_end[1] - edge_start[1]) ** 2)

    # Find box extent along edge direction
    edge_vector = np.array([edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]])
    edge_unit = edge_vector / np.linalg.norm(edge_vector)

    # Project box corners onto edge direction
    projections = []
    for corner in box_corners:
        corner_vector = np.array([corner[0] - edge_start[0], corner[1] - edge_start[1]])
        proj = np.dot(corner_vector, edge_unit)
        projections.append(proj)

    box_min = min(projections)
    box_max = max(projections)

    # Calculate remaining spaces
    space_before = box_min
    space_after = edge_length - box_max

    # Return True if either space is large enough for another car
    return space_before > car_length_depth or space_after > car_length_depth


###BUG### --> In some cases like in clip 8 frame 27 the car width is too big which leeds to an overlap
def boxes_overlap(box1, box2, min_distance=1):
    """
    Check if two boxes overlap or are too close.

    Parameters:
        box1, box2 (list of tuples): Corners of the boxes [(x1, y1), (x2, y2), ...].
        min_distance (int): Minimum distance between any two corners of the boxes.

    Returns:
        bool: True if boxes overlap or are too close, False otherwise.
    """
    def separating_axis_theorem(box1, box2):
        for box in [box1, box2]:
            for i in range(len(box)):
                # Calculate edge vector
                edge = np.array(box[(i + 1) % len(box)]) - np.array(box[i])
                axis = np.array([-edge[1], edge[0]])  # Perpendicular vector
                # Project all points of both boxes onto the axis
                proj1 = [np.dot(np.array(corner), axis) for corner in box1]
                proj2 = [np.dot(np.array(corner), axis) for corner in box2]
                # Check for overlap in projections
                if max(proj1) <= min(proj2) or max(proj2) <= min(proj1):
                    return False  # Found a separating axis or just touching
        return True  # No separating axis found, boxes overlap

    # Check for overlapping interiors
    if separating_axis_theorem(box1, box2):
        return True

    # Check for proximity between corners
    # for corner1 in box1:
    #     for corner2 in box2:
    #         distance = np.linalg.norm(np.array(corner1) - np.array(corner2))
    #         if distance < min_distance:
    #             return True

    return False

def scale_depth(depth_value, depth_min, depth_max, epsilon=1e-6):
    # Function to scale depth values

    depth_value = np.clip(depth_value, depth_min + epsilon, depth_max - epsilon)
    return (depth_max - depth_value) / ((depth_max - depth_min) + epsilon)


def set_markers(line_image, start_point, end_point, vx, vy, disp_resized_np, road_depth_values, previous_boxes, start_depth, end_depth, length_3d):
    # Help Marker Green Edge
    cv2.line(line_image, start_point, end_point, (0, 255, 0), 2)

    angle = np.arctan2(vy, vx)  # Angle in radians
    print(f"Angle: {angle}")

    # Car size without depth
    car_length = 450
    car_width = 240

    amount_cars = length_3d / 450


    # Normalize the depth values
    # depth_min = np.min(disp_resized_np)
    # depth_max = np.max(disp_resized_np)
    depth_min_road = np.min(road_depth_values)
    depth_max_road = np.max(road_depth_values)
    # Use percentiles to exclude outliers
    # depth_min_road = np.percentile(road_depth_values, 1)  # Bottom 5%
    # depth_max_road = np.percentile(road_depth_values, 99)  # Top 95%
    # print(f"Depth Min: {depth_min}, Depth Max: {depth_max}")


    start_depth_normalized = np.log1p(max(start_depth - depth_min_road, 0)) / np.log1p(
        depth_max_road - depth_min_road)
    end_depth_normalized = np.log1p(max(end_depth - depth_min_road, 0)) / np.log1p(depth_max_road - depth_min_road)

    print(f"Road Depth Min: {depth_min_road}")
    print(f"Road Depth Max: {depth_max_road}")
    print(f"Start Depth: {start_depth}")
    print(f"End Depth: {end_depth}")



    # Scale all depth values in the road_depth_values array
    scaled_depths = np.array([
        scale_depth(value, depth_min_road, depth_max_road)
        for value in road_depth_values
    ])

    # Scale start and end depths
    start_depth_scaled = scale_depth(start_depth, depth_min_road, depth_max_road)
    end_depth_scaled = scale_depth(end_depth, depth_min_road, depth_max_road)

    print(f"Scaled Start Depth: {float(start_depth_scaled)}")
    print(f"Scaled End Depth: {float(end_depth_scaled)}")

    # start_depth_normalized = np.clip(start_depth_normalized, 0, 1)
    # end_depth_normalized = np.clip(end_depth_normalized, 0, 1)
    # mean_depth_normalized = (start_depth_normalized + end_depth_normalized) / 2
    start_depth_normalized = start_depth_scaled
    end_depth_normalized = end_depth_scaled

    mean_depth_normalized = (start_depth_scaled + end_depth_scaled) / 2





    print(f"start_depth_normalized: {start_depth_normalized}")
    print(f"end_depth_normalized: {end_depth_normalized}")
    print(f"mean_depth_normalized: {mean_depth_normalized}")

    print(f"start_depth: {start_depth}")
    print(f"end_depth: {end_depth}")


    start_depth_scaled

    car_length_depth = int(450 * mean_depth_normalized)
    car_width_depth_mean = int(240 * mean_depth_normalized)
    if car_width_depth_mean < 50:
        car_width_depth_mean = 50
    car_width_depth_start = int(240 * start_depth_normalized)
    if car_width_depth_start < 50:
        car_width_depth_start = 50
    car_width_depth_end = int(240 * end_depth_normalized)
    if car_width_depth_end < 50:
        car_width_depth_end = 50

    car_length_depth = int(450 * mean_depth_normalized)
    car_width_depth = int(240 * mean_depth_normalized)

    # Update car dimensions
    car_length = int(car_length_depth)
    car_width = int(car_width_depth)
    # car_length = car_length_depth
    deep_position = end_point[0]
    deep_position_2 = start_point[0] + (car_length)*1.1
    more_than_2_cars = False

    corner_2 = (start_point[0] + car_length, start_point[1])
    corner_3 = (start_point[0] + car_length, start_point[1] + car_width)
    angle_active = True

    # Top left 0,0 top right 0,1 bottom right 1,1 bottom left 1,0
    overlap_found = False
    if start_point[0] < end_point[0]:
        if start_point[1] < end_point[1]:
            # Top left
            print('Top left')
            corner_1 = (end_point[0], end_point[1] + car_width_depth_end) # bottom right
            corner_2 = (start_point[0], start_point[1] + car_width_depth_start) # bottom left
            corner_3 = (start_point[0], start_point[1]) # top left
            corner_4 = (end_point[0], end_point[1] ) # top right

            corners = [corner_1, corner_2, corner_3, corner_4]

            if length_3d > (threshold*1.5) and angle_active:
                corner_5 = (deep_position_2, start_point[1])
                corner_6 = (deep_position_2, start_point[1] + (car_width_depth_mean)*0.8)
                corner_5 = rotate_point(corner_5, start_point, angle)
                corner_6 = rotate_point(corner_6, start_point, angle)
                corners.append(corner_5)
                corners.append(corner_6)
                #corners = [rotate_point(corner, start_point, angle) for corner in corners]
                more_than_2_cars = True



    if start_point[0] < end_point[0]:
        if start_point[1] > end_point[1]:
            print('Top right')
            more_than_2_cars = False
            corner_1 = (end_point[0], end_point[1] + car_width_depth_end)  # bottom right
            corner_2 = (start_point[0], start_point[1] + car_width_depth_start)  # bottom left
            corner_3 = (start_point[0], start_point[1])  # top left
            corner_4 = (end_point[0], end_point[1])  # top right

            corners = [corner_1, corner_2, corner_3, corner_4]
####BUG### --> the threshold value if not true as clip_7_frame_8 has a seperation line in the right marker
            if length_3d > (threshold*1.5) and angle_active:
                corner_5 = (deep_position_2, start_point[1])
                corner_6 = (deep_position_2, start_point[1] + (car_width_depth_mean)*0.8)###BUG### --> the value (car_width_depth_mean)*0.8 is not good enough in some cases (clip_8_frame_27)

                corner_5 = rotate_point(corner_5, start_point, angle)
                corner_6 = rotate_point(corner_6, start_point, angle)
                corners.append(corner_5)
                corners.append(corner_6)
                # Only rotate corner 5 and 6


                more_than_2_cars = True








    # Check against previous boxes

    for i in range(0, len(previous_boxes), 4):  # Iterate over groups of 4 corners
        prev_corners = previous_boxes[i:i + 4]

        # Comment out after test
        if overlap_found:
            break

        if boxes_overlap(corners, prev_corners):
            overlap_found = True
            break
    # print(f"§$§$§${len(previous_boxes)}")

    if not overlap_found:
        # Rotate and draw the box if no overlap

        for i in range(4):
            cv2.line(line_image, corners[i], corners[(i + 1) % 4], (0, 0, 255), 2)
        if more_than_2_cars and len(corners) > 4:
            # for i in range(4, 6):
            print(len(corners))
            cv2.line(line_image, corners[4], corners[5], (0, 0, 255), 2)
            print("More than 2 cars")



        # Add the current box to the list of previous boxes
        previous_boxes.extend(corners)
    else:
        print("Overlap detected. Adjust position or size.")


threshold = 10

# return a Scene object with useful methods and image-dependant values


if __name__ == "__main__": 
    
    clip = 7
    frame = 8
    scene = Scene(f'./dataset/sample_data/dataset/annotation_image_action_without_bb/clip_{clip}/{frame}_without_bb.png',
                         f'./dataset/processed_depth/clip_{clip}/{frame}_without_bb_disp.npy',
                         f'./dataset/sample_data/processed/clip_{clip}/{frame}.npy'
                         )