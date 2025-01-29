import cv2
import numpy as np

def mask_to_polygons(binary_mask, interpolation_steps=4):
    height, width = binary_mask.shape
    enlarged_mask = cv2.resize(
        binary_mask.astype(np.uint8), 
        (width * interpolation_steps, height * interpolation_steps),
        interpolation=cv2.INTER_LINEAR
    )
    
    contours, _ = cv2.findContours(
        enlarged_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_float = largest_contour.astype(np.float32)
        
        contour_float /= interpolation_steps
        
        epsilon = 0.005 * cv2.arcLength(contour_float, True)
        approx_polygon = cv2.approxPolyDP(contour_float, epsilon, True)
        polygon_coords = approx_polygon.flatten().tolist()
        
        return polygon_coords
    
    return []