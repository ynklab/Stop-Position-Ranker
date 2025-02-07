import os
import numpy as np
import scipy.ndimage as ndimage
import cv2
from typing import Tuple, List, Any
import math
from shapely.geometry import Polygon
from matplotlib.path import Path


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
    _car_width = 200

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
    
    def __explore(self, starting_point: Tuple[int, int], orientation: float, mask: np.ndarray, length: int) -> Tuple[int, int]:
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
            l = factor * iteration
            if l > length:
                keep_expanding = False
            qx, qy = starting_point[0] + l, starting_point[1]
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

    def __center_square(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
        deltax = (p1[0] - p2[0])/2
        deltay = (p1[1] - p2[1])/2
        return p1[0] + deltax, p1[0] + deltay

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

        # curb/terrain == 2/9; road == 13; sidewalk == 15; pedestrians == 19; crosswalks == 23; lane markings == 24, potholes == 41
        # select pixels with label class == e with e in {road, sidewalk, pedestrian, lane markings, crosswalk} as road
        # we consider pedestrians as road because they might move.
        road_only = np.where((img_mask == 13) | (img_mask == 15) | (img_mask == 23) | (img_mask == 24) | (img_mask == 41), 1, 0)
        # finding 3 biggest clusters of pixels
        lw, _ = ndimage.label(road_only)
        unique, counts = np.unique(lw, return_counts=True)

        is_bounded = lambda x: lambda l: lambda r: l[0] <= x[0] and x[0] <= r[0] and l[1] <= x[1] and x[1] <= r[1]

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
        
        #### EDGE DETECTION
        # borders are: lane markings, curb, terrain, crosswalk, sidewalk, truck, car, other vehicle
        border_points_mask = np.where((img_mask == 2) | (img_mask == 9) | (img_mask == 15) | (img_mask == 23) | (img_mask == 24) | (img_mask == 55)| (img_mask == 59) | (img_mask == 61), 255, 0).astype(np.uint8)
        cv2.imwrite("imgmask.png", border_points_mask)
        # Apply edge detection using Canny
        edges = cv2.Canny(border_points_mask, threshold1=100, threshold2=200)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 25  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 70  # minimum number of pixels making up a line
        max_line_gap = 40.0  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on
        mask_stop_image = np.copy(img)


        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        lines = sorted(list(lines), key=lambda x: x[0][0])
        boxes: List[BoundingBox] = []
        boxid = 0
        polygon_mask = np.zeros(road_only.shape)
        stop_position_segmemts = np.zeros(road_only.shape)

        
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 4)
                number_of_points = max(abs(x1 - x2), abs(y1 - y2))
                xs = np.linspace(x1, x2, number_of_points)
                ys = np.linspace(y1, y2, number_of_points)
                vector = [x2 - x1, y2 - y1]
                orientation = np.arctan2(vector[1], vector[0])
                left_point = (-1, -1)
                right_point = (-1, -1)
                for x, y in zip(xs, ys):
                    point = (x,y)
                    if left_point[0] == -1 or not(is_bounded(point)(left_point)(right_point)):
                        clamped = self.__clamp_point(point, width, height)
                        if polygon_mask[clamped[1], clamped[0]] == 1:
                            continue
                        depth_p = depth[clamped[1], clamped[0]]
                        percentage_p = self.__scale_depth(depth_p, mindepth, maxdepth)
                        
                        scaled_car_width_p = int(self._car_width * percentage_p * 0.9)
                        p_corners = self.__orient_point(clamped, orientation, scaled_car_width_p, road_only)
                        p2_corners = [[-1, -1], [-1, -1]]
                        p2s = [[-1, -1], [-1, -1]]
                        car_ratios = [0,0]
                        index = -1
                        for p_corner in p_corners:
                            index += 1
                            if p_corner[0] == -1:
                                continue
                            vector = [clamped[0] - p_corner[0], clamped[1] - p_corner[1]]
                            width_angle = np.arctan2(vector[1], vector[0])
                            exploration_length = self._car_length * percentage_p * 2.1
                            p2_corner = self.__explore(p_corner, orientation, road_only, exploration_length) 

                            p2x, p2y = p2_corner[0] + scaled_car_width_p, p2_corner[1]
                            p2 = self.__rotate_point((p2x, p2y), p2_corner, width_angle)
                            clamped_p2 = self.__clamp_point(p2, width, height)
                            if clamped_p2[0] != p2[0] or clamped_p2[1] != p2[1]:
                                continue
                            
                            depth_p2 = depth[p2[1], p2[0]]
                            percentage_p2 = self.__scale_depth(depth_p2, mindepth, maxdepth)

                            left_point = (min(clamped[0], p2[0]), min(clamped[1], p2[1]))
                            right_point = (max(clamped[0], p2[0]), max(p2[1], p2[1]))
                            
                            pixel_length = math.sqrt(((clamped[0] - p2[0])**2) + ((clamped[1] - p2[1])**2))
                            scaled_car_length = int(self._car_length * ((percentage_p+percentage_p2)/2))
                            if pixel_length < scaled_car_length or scaled_car_length < 50:
                                continue
                            # found correct points. Store all values
                            car_ratios[index] = pixel_length / scaled_car_length
                            p2_corners[index] = p2_corner
                            p2s[index] = p2

                        for corner, corner_2, point_2, car_ratio in zip(p_corners, p2_corners, p2s, car_ratios):
                            if corner[0] == -1 or corner_2[0] == -1:
                                continue
                            # found a spot, check if width overlap
                            has_no_overlap = self.__all_outside(corner, corner_2, polygon_mask)
                            has_no_overlap_2 = self.__all_outside(clamped, corner, polygon_mask)
                            has_no_overlap_3 = self.__all_outside(clamped, point_2, polygon_mask)
                            has_no_overlap_4 = self.__all_outside(point_2, corner_2, polygon_mask)
                            
                            if not(has_no_overlap) or not(has_no_overlap_2) or not(has_no_overlap_3) or not(has_no_overlap_4):
                                continue
                            
                            vertices = vertices = [clamped, point_2, corner_2, corner]
                            b: BoundingBox = (boxid, car_ratio, vertices)
                            boxes.append(b)
                            cv2.fillConvexPoly(polygon_mask, np.array(vertices), 1)
                            road_only = np.where((road_only == 1) & (polygon_mask == 0), 1, 0)
                            boxid += 1
        
        # cv2.imwrite("img0.png", line_image)
        # cv2.imwrite("masks.png", polygon_mask * 255)
        # cv2.imwrite("road.png", road_only * 255)
        if len(boxes) == 0:
            return stop_position_segmemts, len(boxes)
        if len(boxes) > 6:
            centers = [(id, self.__center_square(b[0], b[1]), cr) for id, cr, b in boxes]
            center_dist = [(id, sum([math.sqrt(((c[0] - c2[0])**2) + ((c[1] - c2[1])**2)) for _, c2, _ in centers]), cr) for id, c, cr in centers]
            center_dist.sort(key=lambda x: x[1] * (x[2] * .75), reverse=True)
            to_be_removed = [id for id,_, _ in center_dist[6:]]
            bb = []
            for id, ratio, b in boxes:
                if id in to_be_removed:
                    continue
                bb.append((id, ratio, b))
            boxes = bb



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
                break

        # visualization of stop position candidates
        font = cv2.FONT_HERSHEY_SIMPLEX
        id = 1
        for box in boxes:
            _, _, b = box
            polygon = Polygon(b)
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exterior = int_coords(polygon.exterior.coords)
            exterior[:, [1, 0]] = exterior[:, [0, 1]]
            nx, ny = stop_position_segmemts.shape

            # Create vertex coordinates for each grid cell...
            # (<0,0> is at the top left of the grid in this system)
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()

            points = np.vstack((x,y)).T

            path = Path(exterior)
            mask = path.contains_points(points)
            mask = np.transpose(mask.reshape((ny,nx)))
            stop_position_segmemts[mask] = id

            # # mask_stop_image = overlay(mask_stop_image, start_point, end_point, end_corner, start_corner, color=colors[boxid], alpha=0.3)
            # tmp_copy = np.copy(mask_stop_image)
            # cv2.fillPoly(tmp_copy, exterior, color=colors[id])
            # mask_stop_image = cv2.addWeighted(mask_stop_image, 0.6, tmp_copy, 0.4, 0)
            # center = polygon.centroid
            # mask_stop_image = cv2.putText(mask_stop_image,str(id+1),(int(center.x),int(center.y)), font, 3,(255,255,255),5,cv2.LINE_AA)
            id += 1

        return stop_position_segmemts, len(boxes)


