import numpy as np
from scipy.ndimage import measurements
import cv2


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
clip = 7
frame = 8
image_mask = np.load(f'C:/Users/benru/PycharmProjects/Stop-position-ranker/Data/sample_data/processed/clip_{clip}/{frame}.npy')
img = cv2.imread(f'C:/Users/benru/PycharmProjects/Stop-position-ranker/Data/sample_data/dataset/annotation_image_action_without_bb/clip_{clip}/{frame}_without_bb.png')

# select pixels with label class as road
road_only = np.where(image_mask == 0, 1, 0)

#finding 3 biggest clusters of pixels
lw, num = measurements.label(road_only)
unique, counts = np.unique(lw, return_counts=True)
ind = np.argpartition(-counts, kth=4)[:4]
road_after_filter = np.zeros(np.shape(lw))
for cluster_id in unique[ind]:
    if cluster_id != 0:
        road_after_filter = np.where(lw == cluster_id, 1, 0)



sizes = np.shape(lw)
ego_coordinates = [sizes[0], sizes[1]/2]
clusters_after_filter = np.zeros(sizes)
coordinate_farthest_y = -1
for y in range(sizes[1]):
    if coordinate_farthest_y == -1:
        if np.sum(lw[:, y]) > 0:
            coordinate_farthest_y = y
            break
coordinate_farthest_x = -1
for x in range(ego_coordinates[0]):
    if not coordinate_farthest_x != -1:
        if np.sum(lw[x][:]) > 0:
            coordinate_farthest_x = x
            break

horizontal = False

#print(f"coordinate_farthest_y: {coordinate_farthest_y} coordinate_farthest_x: {coordinate_farthest_x}")



# Calculate the mean x and y coordinates
indices = np.argwhere(lw == 1)
mean_x, mean_y = np.mean(indices, axis=0)

# Determine orientation based on the mean values
if mean_x > mean_y:
    horizontal = True
    print("Horizontal")
else:
    print("Vertical")



if horizontal:
    # Deviding the combined pixels into 6 different clusters based on distance from ego horizontally
    sizes = np.shape(lw)
    ego_coordinates = [sizes[0], sizes[1]/2]
    clusters_after_filter = np.zeros(sizes)
    coordinate_farthest = -1
    for x in range(ego_coordinates[0]):
        if not coordinate_farthest != -1:
            if np.sum(lw[x][:]) > 0:
                coordinate_farthest = x
                break
    for x in range(coordinate_farthest,ego_coordinates[0]):
        label_id = int(((x - coordinate_farthest)*3)/(ego_coordinates[0] - coordinate_farthest)) + 1
       # print(f"Label id: {label_id} x: {x} coordinate_farthest: {coordinate_farthest} ego_coordinates: {ego_coordinates[0]}")

        for y in range(0,sizes[1]):
            if lw[x][y] == 1:
                if y > ego_coordinates[1]:
                    clusters_after_filter[x][y] = 2*label_id
                else:
                   clusters_after_filter[x][y] = 2*label_id - 1


else:



    # Define sizes and ego coordinates
    sizes = np.shape(lw)
    ego_coordinates = [sizes[0], sizes[1] / 2]
    clusters_after_filter = np.zeros(sizes)
    coordinate_farthest = -1

    # Initialize y_values array to store the farthest y-coordinate for each row (x)
    y_values = np.zeros(sizes[0], dtype=int)

    # Loop through each row x and store the highest y value for each x
    for x in range(sizes[0]):
        # Temporary variable to track the farthest y-coordinate in the current row

        y_value_temp = -1

        # Loop through each column y
        for y in range(sizes[1]):
            if lw[x][y] == 1:
                y_values[x] = y
                if y > coordinate_farthest_y:
                    coordinate_farthest_y = y


    for y in range(sizes[1]):
        if coordinate_farthest == -1:
            if np.sum(lw[:, y]) > 0:
                coordinate_farthest = y
                break
    coordinate_farthest_x = -1
    for x in range(ego_coordinates[0]):
        if not coordinate_farthest_x != -1:
            if np.sum(lw[x][:]) > 0:
                coordinate_farthest_x = x
                break
    mid_way = (ego_coordinates[1] + coordinate_farthest_x) // 2
   # print(f"mid_way: {mid_way} ego_coordinates: {ego_coordinates[1]} coordinate_farthest: {coordinate_farthest_x}")

    for y in range(coordinate_farthest, sizes[1]):
        vertical_label = int(((y - coordinate_farthest) * 3) / (sizes[1] - coordinate_farthest)) + 1
        for x in range(sizes[0]):
            if lw[x][y] == 1:

                if x < mid_way:  # Lower half
                    # Use even labels (2, 4, 6) for lower section clusters
                    clusters_after_filter[x][y] = 2 * vertical_label
                else:  # Upper half
                    # Use odd labels (1, 3, 5) for upper section clusters
                    clusters_after_filter[x][y] = 2 * vertical_label - 1



# print unique values in the image_mask
#print(f"Unique values in image_mask: {np.unique(image_mask)}")

# Create a ranking of the stopping positions based on the distance_to_ego_y and the boolean pedestrian in the cluster
ranking = []

for i in range(1, 7):
    road_after_filter_sub = np.where(clusters_after_filter == i, 1, 0)
    y_coords, x_coords = np.nonzero(road_after_filter_sub)

    if len(x_coords) > 0 and len(y_coords) > 0:
        # Check entire cluster area for pedestrians
        cluster_area = image_mask[y_coords, x_coords]
        contains_pedestrian = 1 if 11 in cluster_area else 0
        # print the contained numbers in the cluster_area
        print(f"Cluster {i} contains: {np.unique(cluster_area)}")

        # Calculate distance to ego
        start_point = (x_coords[0], y_coords[0])
        distance_to_ego_y = abs(start_point[1] - ego_coordinates[1])
        distance_to_ego_x = abs(start_point[0] - ego_coordinates[0])

        ranking.append((i, distance_to_ego_y, contains_pedestrian))



        print(f"Distance to ego x: {distance_to_ego_x} Distance to ego y: {distance_to_ego_y} Contains pedestrian: {contains_pedestrian}")

# Print the ranking
# Sort the ranking based on if the cluster contains a pedestrian and the distance to ego
ranking.sort(key=lambda x: x[2], reverse=True)
ranking.sort(key=lambda x: x[1])
print("Ranking based on distance to ego:")
for i, (cluster, distance, contains_pedestrian) in enumerate(ranking, 1):
    print(f"Rank {i}: Cluster {cluster}- Contains Pedestrian {contains_pedestrian} - Distance to ego: {distance}")


colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
image_with_masks = np.copy(img)
for i in range(1,7):
    road_after_filter_sub = np.where(clusters_after_filter == i, 1, 0)
    image_with_masks = overlay(image_with_masks, road_after_filter_sub, color=colors[i-1], alpha=0.3)
i
cv2.imwrite(f'sample_with_masks_clusters_clip_{clip}_frame_{frame}_horizontal_or_vertical.png', image_with_masks)



