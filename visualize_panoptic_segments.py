import numpy as np
from scipy.ndimage import measurements
import cv2
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json
from scipy import ndimage

stuff_classes=[
"Bird", # need not be considered
"Ground Animal", # need not be considered
"Curb",
"Fence",
"Guard Rail",
"Barrier",
"Wall",
"Bike Lane",
"Crosswalk - Plain",
"Curb Cut", # need not be considered
"Parking",
"Pedestrian Area",
"Rail Track",# need not be considered
"Road",
"Service Lane",# need not be considered
"Sidewalk",
"Bridge",# need not be considered
"Building",# need not be considered
"Tunnel",# need not be considered
"Person",
"Bicyclist",
"Motorcyclist",
"Other Rider",
"Lane Marking - Crosswalk",
"Lane Marking - General",
"Mountain",# need not be considered
"Sand",# need not be considered
"Sky",# need not be considered
"Snow",# need not be considered
"Terrain",# need not be considered
"Vegetation",
"Water",# need not be considered
"Banner",
"Bench",
"Bike Rack",
"Billboard",# need not be considered
"Catch Basin",# need not be considered
"CCTV Camera",# need not be considered
"Fire Hydrant",
"Junction Box",
"Mailbox",
"Manhole",
"Phone Booth",
"Pothole",
"Street Light",
"Pole",
"Traffic Sign Frame",
"Utility Pole",
"Traffic Light",
"Traffic Sign (Back)",
"Traffic Sign (Front)",
"Trash Can",
"Bicycle",
"Boat",
"Bus",
"Car",
"Caravan",
"Motorcycle",
"On Rails",# need not be considered
"Other Vehicle",
"Trailer",
"Truck",
"Wheeled Slow",# need not be considered
"Car Mount",# need not be considered
"Ego Vehicle",
"Stop Positions"
]

_COLORS = []

def gen_color():
    color = tuple(np.round(np.random.choice(range(256), size=3), 3))
    if color not in _COLORS and np.mean(color) != 0.0:
        _COLORS.append(color)
    else:
        gen_color()


for _ in range(300):
    gen_color()

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

image_mask = np.load("scene_graph_batch-2/panoptic_scene_graph_batch_2/clip_2/000480_panoptic_updated.npy")
img = cv2.imread('dataset/images_anonymized/clip_2/images/000480.png')

image_with_masks = np.copy(img)
for i in range(1,np.max(image_with_masks)):
    road_after_filter_sub = np.where(image_mask == i, 1, 0)
    image_with_masks = overlay(image_with_masks, road_after_filter_sub, color=_COLORS[i-1], alpha=0.5)
# cv2.imwrite('scene_graph_explanation.png', image_with_masks)

# img = cv2.imread('scene_graph_explanation.png')
with open("scene_graph_batch-2/panoptic_scene_graph_batch_2/clip_2/000480_panoptic_segments_updated.json", 'r') as f:
    segments_info = json.load(f)

for segment in segments_info:
    binary_mask = np.where(image_mask == segment["id"], 1, 0)
    text = stuff_classes[segment["category_id"]] + "_" + str(segment["id"])
    if np.max(binary_mask) == 0:
        continue
    else:
        # draw text on the largest component
        lw, num = measurements.label(binary_mask)
        unique, counts = np.unique(lw, return_counts=True)
        sorted_unique = [x for _, x in sorted(zip(counts, unique))]
        sorted_unique_check = [[x,count] for count, x in sorted(zip(counts, unique))]
        try:
            max_cluster = sorted_unique[-2]
        except:
            max_cluster = sorted_unique[-1]
        lw[lw != max_cluster] = 0 
        x, y = ndimage.center_of_mass(lw)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_with_masks,text,(int(y),int(x)), font, 0.5,(255,255,255),2,cv2.LINE_AA)

cv2.imwrite('scene_graph_explanation_with_labels.png', image_with_masks)



