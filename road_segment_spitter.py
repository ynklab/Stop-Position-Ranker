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

image_mask = np.load("processed/clip_5/21.npy")
img = cv2.imread('dataset/annotation_image_action_without_bb/clip_5/21_without_bb.png')


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

# Deviding the combined pixels into 6 different clusters based on distance from ego
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
    for y in range(0,sizes[1]):
        if lw[x][y] == 1:
            if y > ego_coordinates[1]:
                clusters_after_filter[x][y] = 2*label_id
            else:
               clusters_after_filter[x][y] = 2*label_id - 1 

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
image_with_masks = np.copy(img)
for i in range(1,7):
    road_after_filter_sub = np.where(clusters_after_filter == i, 1, 0)
    image_with_masks = overlay(image_with_masks, road_after_filter_sub, color=colors[i-1], alpha=0.3)
cv2.imwrite('sample_with_masks_clusters.png', image_with_masks)



