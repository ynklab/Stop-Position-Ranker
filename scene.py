import os
import numpy as np
import scipy.ndimage as ndimage
import cv2
from typing import Tuple, List, Any
import math
BoundingBox = Tuple[int, int, List[Tuple[int, int]]]
colors = [
    (0,0,255),
    (0,255,0),
    (255,0,0),
    (255,255,0),
    (255,0,255),
    (0,255,255)
]

class Scene():
    _car_length = 250
    _car_width = 160

    def __init__(self, img_path: str, depth_map_path: str, mask_npy_path: str):
        assert os.path.exists(img_path), "input image was not found"
        assert os.path.exists(depth_map_path), "depth map .npy file was not found"
        assert os.path.exists(mask_npy_path), "mask path not found"
        self._img_path, self._depth_map_path, self._mask_npy_path = img_path, depth_map_path, mask_npy_path

    def __clamp_point(self, point: List[int], width: int, height: int) -> Tuple[int, int]:
        """
        Forces a point to be within [:width, :height]
        """
        x, y = point
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return int(x), int(y)

    def __rotate_point(self, p: Tuple[int, int], origin: Tuple[int, int], angle: float) -> Tuple[int, int]:
        """
        rotates a point around a given origin and angle
        """
        ox, oy = origin
        px, py = p
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy) 
        qx, qy = int(qx), int(qy)
        return qx, qy


    def __orient_point(self, p: Tuple[int, int], orientation: float, dist: int, mask: np.ndarray, straight = False) -> List[Tuple[int, int]]:
        """
        reorients a point, wrt a given angle (orientation) and a distance (width and length)
        only returns points on a road mask, (-1, -1) otherwise
        could (should) be recursive.
        """
        # pi/2 or 3pi/2, depending on the direction of the road 
        right_angle = orientation
        result = [[-1, -1], [-1, -1]]
        if not straight:
            right_angle = orientation + (math.pi/2)
        ox, oy = p
        px, py = p
        px += dist
        qx, qy = self.__rotate_point((px, py), (ox, oy), right_angle)
  
        clamped_point = self.__clamp_point((qx, qy), mask.shape[1], mask.shape[0])
        is_clamped = all([i == j for i, j in zip((qx, qy), clamped_point)])
        if is_clamped and mask[qy, qx] == 1:
 
            result[0] = (qx, qy)
        if straight:
            return result
        right_angle = right_angle + math.pi
        qx, qy = self.__rotate_point((px, py), (ox, oy), right_angle)
        is_clamped = all([i == j for i, j in zip((qx, qy), self.__clamp_point((qx, qy), mask.shape[1], mask.shape[0]))])
        if is_clamped and mask[qy, qx] == 1:
            result[1] = (qx, qy)
        return result
    
    def __explore(self, starting_point: Tuple[int, int], orientation: float, mask: np.ndarray) -> Tuple[int, int]:
        """
        Returns the point which 
        1. is the farthest from starting point
        2. is on the road
        3. follows the line provided by orientation
        """
        keep_expanding = True
        factor = 5
        iteration = 0
        px, py = starting_point 
        while keep_expanding:
            iteration += 1
            qx, qy = starting_point[0] + (factor * iteration), starting_point[1]
            qx, qy = self.__rotate_point((qx, qy), starting_point, orientation)
            is_clamped = all([i == j for i, j in zip((qx, qy), self.__clamp_point((qx, qy), mask.shape[1], mask.shape[0]))])
            if is_clamped and mask[qy, qx] == 1:
                px, py = qx, qy
            else:
                keep_expanding = False
        return px, py
    

    def __find_first(self, starting_point: Tuple[int, int], limit: Tuple[int, int], orientation: float, mask: np.ndarray) -> Tuple[int, int]:
        """
        Basically the opposite condition of explore. Find the point which:
        1. is the closest from starting point
        2. is outside the mask
        3. follow the line provided by orientation
        4. stops at a limit
        returns (-1,-1) if nothing is found
        """
        col_sign = 1 if starting_point[0] < limit[0] else -1
        row_sign = 1 if starting_point[1] < limit[1] else -1
        qx, qy = -1, -1
        uninterrupted = False
        for col in range(starting_point[0], limit[0], 1*col_sign):
            for row in range(starting_point[1], limit[1], 1*row_sign):
                if not uninterrupted:
                    if mask[row, col] != 1.0:
                        qx, qy = col, row
                        uninterrupted = True
                else:
                    if mask[row, col] == 1.0:
                        qx, qy = -1, -1
                        uninterrupted = False
                    
        return qx, qy

    def __all_outside(self, starting_point: Tuple[int, int], limit: Tuple[int, int], mask: np.ndarray) -> bool:
        """
        These 3 methods should be unified, with a boolean condition as a parameter. Until then:
        checks if all points between starting_point and limit are outside the given mask.
        """
        m = np.zeros(mask.shape)
        minimum = min(starting_point[0], limit[0]), min(starting_point[1], limit[1])
        maximum = max(starting_point[0], limit[0]), max(starting_point[1], limit[1])
        m[minimum[1]:maximum[1], minimum[0]:maximum[0]] = mask[minimum[1]:maximum[1], minimum[0]:maximum[0]]
        return not m.any()
                

 
    def __scale_depth(self, depth_value, depth_min, depth_max, epsilon=1e-6):
        # Function to scale depth values

        depth_value = np.clip(depth_value, depth_min + epsilon, depth_max - epsilon)
        return (depth_max - depth_value) / ((depth_max - depth_min) + epsilon)


    def process_image(self) -> np.ndarray:
        """
        Returns a list of bounding boxes b.
        Each bounding box is a list of 4 tuples t, where t = Tuple[int, int]
        """
        # we should have asserts to validate that given str is os.path
        #### DATA LOADING
        img = cv2.imread(self._img_path)
        img_mask = np.load(self._mask_npy_path)
        depth = np.load(self._depth_map_path, allow_pickle=True).squeeze()
        height, width = img.shape[0], img.shape[1]

        ##### DEPTH MAPPING OF ROAD ONLY
        # select pixels with label class as road
        road_only = np.where(img_mask == 0, 1, 0)

        # finding 3 biggest clusters of pixels
        lw, _ = ndimage.label(road_only)
        unique, counts = np.unique(lw, return_counts=True)
        # assumes that background is always the biggest cluster
        if len(counts) > 4:
            ind = np.argpartition(-counts, kth=4)[:4]
            biggest_clusters_index = list(unique[ind])

            # index 0 is background
            if 0 in biggest_clusters_index:
                biggest_clusters_index.remove(0)
        else:
            biggest_clusters_index = unique[1:]
        road_after_filter = np.array([[lw[y, x] in biggest_clusters_index for x in range(width)] for y in range(height)])
        
        # Extract depth values only for road pixels
        road_depth_values = np.where(road_after_filter, depth, 0)
        mindepth = np.min(road_depth_values[np.nonzero(road_depth_values)])
        maxdepth = np.max(road_depth_values)
        #road_mask = np.where(road_after_filter)

        #### EDGE DETECTION
        border_points_mask = np.where(img_mask == 1, 255, 0).astype(np.uint8)


        # Apply edge detection using Canny
        edges = cv2.Canny(border_points_mask, threshold1=100, threshold2=200)

        # Find contours (edge points)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = list(contours)
        
        line_image = np.copy(img)
        boxes: List[BoundingBox] = []
        boxid = 0
        polygon_mask = np.zeros(road_only.shape)
        for contour in contours:
            if len(contour) < 500:  # Skip very small contours
                continue
            # Fit a straight line to the entire contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Project contour points onto the line
            projections = []
            for point in contour:
                px, py = point[0]
                # Parametric projection of the point onto the line
                t = vx * (px - x) + vy * (py - y)
                projections.append([x + t * vx, y + t * vy])
            projections = np.array(projections)

            # Find the extreme points along the line
            min_t_idx = np.argmin(projections[:, 0] * vx + projections[:, 1] * vy)
            max_t_idx = np.argmax(projections[:, 0] * vx + projections[:, 1] * vy)

            # line orientation
            orientation = np.arctan2(vy, vx)
            orientation %= (math.pi*2)
            # Convert to integer and clamp within bounds
            start_point = self.__clamp_point(projections[min_t_idx], width, height)
            end_point = self.__clamp_point(projections[max_t_idx], width, height)
            if polygon_mask[start_point[1], start_point[0]]:
                start_point = self.__find_first(start_point, end_point, math.pi - orientation, polygon_mask )
                if start_point[0] == -1:
                    continue
            if polygon_mask[end_point[1], end_point[0]]:
                end_point = self.__find_first(end_point, start_point, orientation, polygon_mask )
                if end_point[0] == -1:
                    continue

            #cv2.line(line_image, start_point, end_point, (0,255,0), 4)

            # euclidian distance, pixelwise
            pixel_length = math.sqrt(((end_point[0] - start_point[0])**2) + ((end_point[1] - end_point[1]) ** 2)) 

            # Access depth values at valid indices
            start_depth = depth[start_point[1], start_point[0]]
            end_depth = depth[end_point[1], end_point[0]]

            percentage_start = self.__scale_depth(start_depth, mindepth, maxdepth*1.2)
            percentage_end = self.__scale_depth(end_depth, mindepth, maxdepth*1.2)
            scaled_car_length = int(self._car_length * ((percentage_start+percentage_end)/2))
            if pixel_length < scaled_car_length:
                continue 
            scaled_car_width_start = int(self._car_width * percentage_start * 0.9)
            scaled_car_width_end = int(self._car_width * percentage_end * 0.9)

            start_corners = self.__orient_point(start_point, orientation, scaled_car_width_start, road_only)
            end_corners = self.__orient_point(end_point, orientation, scaled_car_width_end, road_only)
            index = -1
            for start_corner, end_corner in zip(start_corners, end_corners):
                index += 1 
                if start_corner[0] == -1 and end_corner[0] == -1:
                    continue # neither points are on the road
                elif end_corner[0] == -1: # end corner not on the road
                    # try to find minimal bb
                    points = self.__orient_point(start_corner, orientation, scaled_car_length, road_only, True)
                    qx, qy = points[0]
                    if qx == -1:
                        continue
                    # minimal bounding box can be on the road. Explore to expand it
                    end_corner = self.__explore((qx, qy), orientation, road_only)
                    vector = [start_point[0] - start_corner[0], start_point[1] - start_corner[1]]
                    correct_angle = np.arctan2(vector[1], vector[0])
                    # find the accurate end_point
                    px, py = end_corner[0] + scaled_car_width_start, end_corner[1]
                    end_point = self.__rotate_point((px, py), end_corner, correct_angle)
                    end_corners[index] = end_corner
                elif end_corner == 0:
                    angle = math.pi + (orientation)
                    angle %= (2*math.pi)
                    points = self.__orient_point(end_corner, angle, scaled_car_length, road_only, True)
                    qx, qy = points[0]
                    if qx == -1:
                        continue
                    start_corner = self.__explore((qx, qy), math.pi - orientation, road_only)
                    vector = [end_point[0] - end_corner[0], end_point[1] - end_corner[1]]
                    correct_angle = np.arctan2(vector[1], vector[0])
                    # find the accurate start_point
                    px, py = start_corner[0] + scaled_car_width_end, start_corner[1]
                    start_point = self.__rotate_point((px, py), start_corner, correct_angle)
                    start_corners[index] = start_corner

            for start_corner, end_corner in zip(start_corners, end_corners):
                if start_corner[0] == -1 or end_corner[0] == -1:
                    continue
                # found a spot here, check if overlap
                has_no_overlap = self.__all_outside(start_corner, end_corner, polygon_mask)
                if not has_no_overlap:
                    continue
            
                vertices = [start_point, end_point, end_corner, start_corner]
                car_ratio = pixel_length / scaled_car_length
                b: BoundingBox = (boxid, car_ratio, vertices)
                boxes.append(b)
                cv2.fillConvexPoly(polygon_mask, np.array(vertices), 1)
                # all time is lost here
                road_only = np.where((road_only == 1) & (polygon_mask == 0), 1, 0)
                
                boxid += 1
        if len(boxes) == 0:
            return line_image, 0
        while len(boxes) > 6:
            boxes.pop()
        while len(boxes) < 6:
            big_id, car_ratio, biggest = boxes[0]
            start_point, end_point, end_corner, start_corner = biggest
            biggest_length = math.sqrt(((end_point[0] - start_point[0])**2) + ((end_point[1] - end_point[1]) ** 2)) 
            for id, cr, b in boxes[1:]:
                s2, e2, _, _ = b
                l = math.sqrt(((e2[0] - s2[0])**2) + ((e2[1] - s2[1]) ** 2))
                if l > biggest_length:
                    biggest_length = l
                    big_id = id
                    car_ratio = cr 
                    start_point, end_point, end_corner, start_corner = b
            if car_ratio >= 2:
                middle_point_x = int((start_point[0]+end_point[0])/2)
                middle_point_y = int((start_point[1]+end_point[1])/2)
                middle_corner_x = int((start_corner[0]+end_corner[0])/2)
                middle_corner_y = int((start_corner[1]+end_corner[1])/2)
                boxes[big_id] = (big_id, car_ratio/2, [(middle_point_x, middle_point_y), end_point, end_corner, (middle_corner_x, middle_corner_y)])
                boxes.append((len(boxes), car_ratio/2, [start_point, (middle_point_x, middle_point_y), (middle_corner_x, middle_corner_y), start_corner]))
            else: # cannot make more bounding boxes
                print("Could not find 6 bounding boxes in this image")
                break
        for box in boxes:
            boxid, _, b = box
            start_point, end_point, end_corner, start_corner = b
            cv2.line(line_image, start_corner, end_corner, colors[boxid], 2)
            cv2.line(line_image, start_corner, start_point, colors[boxid], 2)
            cv2.line(line_image, end_point, end_corner, colors[boxid], 2)
            cv2.line(line_image, start_point, end_point, colors[boxid], 2)

        return line_image, len(box)


