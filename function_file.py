import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage import color
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def classify_pole_color(img_rgb, binary_mask, binary_mask_name, folder_title, visualize=False):
    mask_bool = binary_mask > 0
    extracted_pixels = img_rgb[mask_bool]
    pixels_reshaped = extracted_pixels.reshape(1, -1, 3)
    lab_pixels = color.rgb2lab(pixels_reshaped).reshape(-1, 3)
    a_channel = lab_pixels[:, 1]
    b_channel = lab_pixels[:, 2]  
    hue_angles_rad = np.arctan2(b_channel, a_channel)
    hue_angles_deg = np.degrees(hue_angles_rad)
    hue_angles_norm = hue_angles_deg % 360
    counts, bin_edges = np.histogram(hue_angles_norm, bins=360, range=(0, 360))
    smoothed_counts = gaussian_filter1d(counts, sigma=3, mode='wrap')
    

    min_height = np.max(smoothed_counts) * 0.15 
    peaks, properties = find_peaks(smoothed_counts, height=min_height)
    
    if len(peaks) == 0:
        return "Unknown"
        

    primary_peak_idx = peaks[np.argmax(properties['peak_heights'])]
    peak_angle = float(primary_peak_idx) 
    peak_height = smoothed_counts[primary_peak_idx]
    

    half_max = peak_height / 2.0
    

    left_bound = primary_peak_idx
    while smoothed_counts[left_bound] > half_max:
        left_bound = (left_bound - 1) % 360
        if left_bound == primary_peak_idx: break 
        

    right_bound = primary_peak_idx
    while smoothed_counts[right_bound] > half_max:
        right_bound = (right_bound + 1) % 360
        if right_bound == primary_peak_idx: break 
        

    targets = {"Red": 0, "Yellow": 90, "Blue": 270}

    min_dist = float('inf')
    pred_color = "Unknown"
    
    for color_name, target_angle in targets.items():
        dist = min(abs(peak_angle - target_angle), 360 - abs(peak_angle - target_angle))
        if dist < min_dist:
            min_dist = dist
            pred_color = color_name

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.bar(bin_edges[:-1], counts, width=1, color='lightgray')
        plt.plot(np.arange(360), smoothed_counts, color='black', linewidth=2)

        box_colors = {"Red": "red", "Yellow": "yellow", "Blue": "blue", "Unknown": "gray"}
        display_color = box_colors.get(pred_color, 'gray')
        if left_bound <= right_bound:
            plt.axvspan(left_bound, right_bound, color=display_color, alpha=0.3)
        else:
            plt.axvspan(left_bound, 360, color=display_color, alpha=0.3)
            plt.axvspan(0, right_bound, color=display_color, alpha=0.3)
        plt.axvline(x=peak_angle, color='black', linestyle='--', linewidth=2)
        
        plt.xlim(0, 360)
        plt.xticks(np.arange(0, 361, 30))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{folder_title}/mask_name_{binary_mask_name}.png')
        plt.close() 

    return pred_color

def show_anns_img(anns):

    binary_mask = np.zeros((anns[0].shape[0], anns[0].shape[1]), dtype=np.uint8)

    for m in anns:
        binary_mask[m] = 255
    return binary_mask

def boost_underwater_image(img, r_mult=3.0, g_mult=1.0, b_mult=1.0, clip_limit=2.0, grid_size=8):

    b, g, r = cv2.split(img.astype(np.float32))
    
    b = np.clip(b * b_mult, 0, 255)
    g = np.clip(g * g_mult, 0, 255)
    r = np.clip(r * r_mult, 0, 255)
    
    color_corrected = cv2.merge((b, g, r)).astype(np.uint8)
    lab = cv2.cvtColor(color_corrected, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    cl = clahe.apply(l)
    
    merged_lab = cv2.merge((cl, a, b_chan))
    final_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return final_img
def get_yolo_centers(img, model, device):
    original_image = img.copy()
    original_image1 = img.copy()

    results = model.predict(
        source=original_image1,
        conf=0.45,       
        iou=0.5,         
        device=device,
        save=False,      
        show=False       
    )
    
    center_list = []
    
    for result in results:
        boxes = result.boxes  
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist() 
            cx1, cy1 = int((x1+x2)/2), int((y1+y2)/2)
            
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (127,255,255), 4)
            img = cv2.circle(img, (cx1, cy1), 1, (255,127,0), 2)
            
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            center_list.append((cx1, cy1))
            print(f"Detected {class_name} with {conf:.2f} confidence at center [{cx1}, {cy1}]")
            
    return center_list, img, original_image
def apply_sam_masks(original_image, center_list, predictor):

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    

    predictor.set_image(image_rgb)
    
    overlay = original_image.copy()
    color = np.array([255,81,31], dtype=np.uint8)
    all_masks = []
    
    t1 = time.time()

    for (x, y) in center_list:
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]
        all_masks.append(mask)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5 
        cv2.circle(overlay, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        
    print(f"SAM inference for {len(center_list)} points completed in {time.time()-t1:.3f} seconds.")
    
    return all_masks, overlay

