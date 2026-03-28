import sys
import os

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dependencies'))
sys.path.insert(0, deps_path)

import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
import os
import matplotlib
import json


from scipy.signal import find_peaks
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage import color
from ultralytics import YOLO
from depth.depth_anything_v2.dpt import DepthAnythingV2
from scipy.ndimage import gaussian_filter1d
from function_file import classify_pole_color as clasfiy_pole
from function_file import show_anns_img as show_annotations
from function_file import boost_underwater_image as boost_image
from function_file import get_yolo_centers as yolow_centers
from function_file import apply_sam_masks as sam_is_like_that



def inference_is_taken_here (frame_path,folder_title,index):
    global prev_time
    json_objj = {}
    frame = cv2.imread(frame_path,1)
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    frame = cv2.resize(frame, (640, 480))
    raw_image = frame
    raw_image_copy = raw_image.copy()
    raw_image_copy_diff = raw_image.copy()
    raw_image_final = raw_image.copy()

    # ── Depth estimation ──────────────────────────────────────────────────────
    depth = depth_anything.infer_image(raw_image, input_sizee)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_img = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    depth_img_overlay = np.zeros_like(depth_img)

    # ── Optional enhancement ──────────────────────────────────────────────────
    org_img = boost_image(raw_image) if enhance_bool else raw_image


    centers, yolo_annotated_img, orig_img = yolow_centers(depth_img, yolo_model, device)
    masks, sam_overlay_img = sam_is_like_that(orig_img, centers, sam_predictor)
    print('\n sam + yolow + depth took ', time.time()-now, ' seconds \n')

    org_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    combined_mask = np.zeros_like(org_img_rgb)
    combined_bool_mask = np.zeros((org_img_rgb.shape[0], org_img_rgb.shape[1]), dtype=bool)
    bounding_rect_list_with_color = []
    pole_masks = masks 


    # folder_title = "00_master/1"
    # os.makedirs(folder_title, exist_ok=True)

    folder_title = folder_title+f"/__{index}__"
    os.makedirs(folder_title, exist_ok=True)

    for i, mask_array in enumerate(pole_masks):
        single_mask_visual = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        single_mask_visual[mask_array] = 255
        x, y, w, h = cv2.boundingRect(single_mask_visual)
        single_mask_bgr = cv2.cvtColor(single_mask_visual, cv2.COLOR_GRAY2BGR)


        cv2.imwrite(f'{folder_title}/thisismask{i}.png', single_mask_bgr)

        combined_bool_mask = np.logical_or(combined_bool_mask, mask_array)
        combined_mask = cv2.bitwise_or(combined_mask, single_mask_bgr)
        color_value = clasfiy_pole(org_img_rgb, mask_array, f"pole_repo_mask_{i}", folder_title, visualize=True)
        bounding_rect_list_with_color.append(([x, y, w, h], color_value))

    final_img = cv2.bitwise_and(combined_mask, org_img_rgb)
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

    for in_dex, ([x, y, w, h], color_value) in enumerate(bounding_rect_list_with_color):
        raw_image_copy_diff = cv2.rectangle(
            raw_image_copy_diff, (x, y), (x + w, y + h), color_dict[color_value], 4
        )
        json_objj[f"{in_dex}"] = {
            'coords':[x,y,w,h],
            'color':color_value,
            'area':(w*h)
        }


    cv2.imwrite(f'{folder_title}/depth_img.png', depth_img)
    cv2.imwrite(f'{folder_title}/final_img.png', final_img_bgr)
    cv2.imwrite(f'{folder_title}/enhanced_img.png', org_img)
    cv2.imwrite(f'{folder_title}/org_img.png', raw_image_copy)
    cv2.imwrite(f'{folder_title}/predicted_img.png', raw_image_copy_diff)
    cv2.imwrite(f'{folder_title}/depth.png', depth_img)
    clasfiy_pole(org_img_rgb, combined_bool_mask, "fin", folder_title, visualize=True)
    return raw_image_copy,json_objj




sam_path = 'checkpoints/sam_vit_h_4b8939.pth'
yolo_model = YOLO('checkpoints/best.pt')
model_typee = 'vitl'
write_bool = False
sam_bool = False
input_sizee = 518
enhance_bool = True

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

color_dict = {
    "Red":    (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Blue":   (255, 0, 0)
}

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading YOLO model...")
print(f"Loading SAM model on {device}...")
sam = sam_model_registry["vit_h"](checkpoint=sam_path)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,   96,   192,   384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,   192,  384,   768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256,  512,  1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**model_configs[model_typee])
depth_anything.load_state_dict(
    torch.load(f'checkpoints/depth_anything_v2_{model_typee}.pth', map_location='cpu')
)
depth_anything = depth_anything.to(device).eval()

prev_time = time.time()




main_json = {}


folder_patth = r'test_folder'
raw_image_list = []
valid_extensions = ('.png', '.jpg', '.jpeg')
folder_title = datetime.datetime.now().strftime("03_master/0_-%m-%d_%H%M__%Y_%S")
if os.path.exists(folder_patth):
    for index,filename in enumerate(os.listdir(folder_patth)):
        if filename.endswith(valid_extensions):
            full_path = os.path.join(folder_patth, filename)
            imge,json_obj = inference_is_taken_here(full_path,folder_title,index)
            raw_image_list.append(imge)
            main_json[f'{index}'] = json_obj

    os.makedirs(folder_title, exist_ok=True)

    with open(f'{folder_title}/all_results.json', 'w') as f:
        json.dump(main_json, f, indent=4)
    
    
    
    
# with open(f'02_master/0_-03-27_1245__2026_30/all_results.json', 'r') as f:
#     main_json = json.load(f)

#     print(len(main_json.items))


flattened_list = []
print(list(main_json.keys()))
for frame_id,frame_data in main_json.items():
    for obj_id,obj_data in frame_data.items():
        x,y,w,h = obj_data['coords']
        c1 = x+(w)//2
        c2 = y+(h)//2
        flattened_list.append(
            {
            "frame": frame_id,
            "color": obj_data["color"],
            "coords": obj_data["coords"],
            "area": obj_data["area"],
            "center": (c1, c2)
            }
        )

# print(json.dumps(flattened_list,indent=4))
min_distance = 50 

cluster_coords = []

for elem in flattened_list:
    cen_x,cen_y = elem["center"]
    current_cluster = None

    for cluster in cluster_coords:
        print('one_cluster',cluster)
        mean_x = np.mean([singular_cluster['center'][0] for singular_cluster in cluster])
        mean_y = np.mean([singular_cluster['center'][1] for singular_cluster in cluster])
        print('check1')
        dist =  np.sqrt(np.abs(mean_x - cen_x)**2 + np.abs(mean_y - cen_y)**2)
        if dist<min_distance:
            current_cluster = cluster
            break
    if current_cluster is not None:
        current_cluster.append(elem)
    else:
        cluster_coords.append([elem])


    
final_data = {}
for i, group in enumerate(cluster_coords):
    avg_coords = np.mean([d["coords"] for d in group], axis=0).astype(int).tolist()
    avg_area   = int(np.mean([d["area"] for d in group]))
    avg_center = np.mean([d["center"] for d in group], axis=0).astype(int).tolist()
    
    colors = [d["color"] for d in group]
    most_freq_color = max(set(colors), key=colors.count)
    
    final_data[f"pole_{i}"] = {
        "color": most_freq_color,
        "avg_coords": avg_coords,
        "avg_center": avg_center,
        "avg_area": avg_area,
        "num_detections": len(group),
        "seen_in_frames": [d["frame"] for d in group]
    }



n1 = raw_image_list[-1]

for pole,pole_data in final_data.items():
    x,y,w,h = pole_data['avg_coords']
    n1 = cv2.rectangle(n1,(x,y),(x+w,y+h),color_dict[pole_data['color']],1)
cv2.imwrite(f'{folder_title}/final_pred.png', n1)
with open(f'{folder_title}/final_results.json', 'w') as f:
    json.dump(final_data, f, indent=4)


