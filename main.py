import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage import color
import datetime
from ultralytics import YOLO
import os
from depth.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import depthai as dai


from function_file import classify_pole_color as clasfiy_pole
from function_file import show_anns_img as show_annotations
from function_file import boost_underwater_image as boost_image
from function_file import get_yolo_centers as yolow_centers
from function_file import apply_sam_masks as sam_is_like_that


# img_path_org= 'or_rby/my_photo-202_jpg.rf.82938ae192be2f81297cdf7901f75df5.jpg'
sam_path= 'checkpoints/sam_vit_h_4b8939.pth'
yolo_model = YOLO('checkpoints/best.pt')
sam_path = 'checkpoints/sam_vit_h_4b8939.pth'
model_typee = 'vitl'
write_bool = False
sam_bool = False


input_sizee = 518
enhance_bool = True
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
color_dict = {"Red":(0,0,255),
              "Yellow":(0,255,255),
              "Blue":(255,0,0)
              }
## this is loading of modelss
print("Loading YOLO model...")
print(f"Loading SAM model on {device}...")
sam = sam_model_registry["vit_h"](checkpoint=sam_path)
sam.to(device=device)
sam_predictor = SamPredictor(sam)


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(DEVICE)
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


depth_anything = DepthAnythingV2(**model_configs[model_typee])
depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{model_typee}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()





pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
# cam.setPreviewSize(640, 360)   
cam.setPreviewSize(800, 450)     
# cam.setPreviewSize(800, 600)       # Fast preview
cam.setInterleaved(False)
cam.setFps(25)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Start OAK device
device = dai.Device(pipeline)
q_video = device.getOutputQueue("video", maxSize=1, blocking=False)
# q_video = device.getOutputQueue("video")
###############

prev_time = time.time()


while True:
    now = time.time()
    in_frame = q_video.get()
    frame = in_frame.getCvFrame()
    # FPS calculation 
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    frame = cv2.resize(frame,(640,480))

    filenames = frame 


    raw_image = cv2.imread(filenames)
    raw_image_copy = raw_image.copy()
    raw_image_copy_diff = raw_image.copy()

    depth = depth_anything.infer_image(raw_image,input_sizee)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    depth_img = depth

    depth_img_overlay = np.zeros_like((depth_img)) 

    org_img = raw_image
    if enhance_bool:
        org_img = boost_image(raw_image)

    centers, yolo_annotated_img, orig_img = yolow_centers(depth_img, yolo_model, device)

    if sam_bool:
        masks, sam_overlay_img = sam_is_like_that(orig_img, centers, sam_predictor)
    else:
        masks = [] 
        sam_overlay_img = org_img

    org_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    combined_mask = np.zeros_like(org_img_rgb)
    combined_bool_mask = np.zeros((org_img_rgb.shape[0], org_img_rgb.shape[1]), dtype=bool)
    bounding_rect_list_with_color = []

    folder_title = "00_master/1"
    os.makedirs(folder_title, exist_ok=True)
    if write_bool:
        folder_title = datetime.datetime.now().strftime("00_master/0_-%m-%d_%H%M__%Y_%S")
        os.makedirs(folder_title, exist_ok=True)
    for i, mask_array in enumerate(pole_masks):
        x,y,w,h = 0,0,0,0
        single_mask_visual = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        single_mask_visual[mask_array] = 255
        x,y,w,h = cv2.boundingRect(single_mask_visual)
        single_mask_bgr = cv2.cvtColor(single_mask_visual, cv2.COLOR_GRAY2BGR)
        if write_bool:
            cv2.imwrite(f'{folder_title}/thisismask{i}.png', single_mask_bgr)
        combined_bool_mask = np.logical_or(combined_bool_mask, mask_array)
        combined_mask = cv2.bitwise_or(combined_mask, single_mask_bgr)
        color_value = clasfiy_pole(org_img_rgb, mask_array, f"pole_repo_mask_{i}", folder_title, visualize=True)
        bounding_rect_list_with_color.append(([x,y,w,h],color_value))


    final_img = cv2.bitwise_and(combined_mask, org_img_rgb)
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)


    for [x,y,w,h],color_value in bounding_rect_list_with_color:
        raw_image_copy_diff = cv2.rectangle(raw_image_copy_diff,(x,y),((x+w),(y+h)),color_dict[color_value],4)

    
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(org_img, f"FPS: {fps:.2f}", (10,30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    if write_bool:
        cv2.imwrite(f'{folder_title}/depth_img.png', depth_img) 
        cv2.imwrite(f'{folder_title}/depth_img_overlay.png', show_annotations(pole_masks)) 
        cv2.imwrite(f'{folder_title}/final_img.png', final_img_bgr)
        cv2.imwrite(f'{folder_title}/enhanced_img.png', org_img) 
        cv2.imwrite(f'{folder_title}/org_img.png', raw_image_copy)
        cv2.imwrite(f'{folder_title}/predicted_img.png', raw_image_copy_diff)
        cv2.imwrite(f'{folder_title}/depth.png', depth)
        clasfiy_pole(org_img_rgb, combined_bool_mask, "fin", folder_title, visualize=True)



