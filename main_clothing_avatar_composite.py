import cv2
import numpy as np
from PIL import Image
import os

def segment_and_crop_clothing(clothing_path):
    """
    Segment clothing from the background and crop it.
    Returns the path to the processed clothing image.
    """
    try:
        # Read the image
        img = cv2.imread(clothing_path)
        if img is None:
            return None

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour (assuming it's the clothing)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Apply mask to original image
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        # Save the processed image
        output_path = "input/processed_clothing.png"
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        return output_path
        
    except Exception as e:
        print(f"Error in clothing segmentation: {str(e)}")
        return None

def composite_images(avatar_path, clothing_path, output_path):
    """
    Composite the clothing onto the avatar.
    """
    try:
        # Read images
        avatar = cv2.imread(avatar_path)
        clothing = cv2.imread(clothing_path)
        
        if avatar is None or clothing is None:
            raise ValueError("Could not read one or both images")
            
        # Resize clothing to fit avatar (adjust size as needed)
        clothing = cv2.resize(clothing, (avatar.shape[1] // 2, avatar.shape[0] // 2))
        
        # Create a mask for the clothing
        gray = cv2.cvtColor(clothing, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Calculate position (center of avatar)
        x_offset = (avatar.shape[1] - clothing.shape[1]) // 2
        y_offset = (avatar.shape[0] - clothing.shape[0]) // 2
        
        # Create ROI
        roi = avatar[y_offset:y_offset+clothing.shape[0], x_offset:x_offset+clothing.shape[1]]
        
        # Create inverse mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Black out the area of clothing in ROI
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        
        # Take only region of clothing from clothing image
        clothing_fg = cv2.bitwise_and(clothing, clothing, mask=mask)
        
        # Put clothing in ROI and modify the main image
        dst = cv2.add(roi_bg, clothing_fg)
        avatar[y_offset:y_offset+clothing.shape[0], x_offset:x_offset+clothing.shape[1]] = dst
        
        # Save the result
        cv2.imwrite(output_path, avatar)
        
    except Exception as e:
        print(f"Error in image compositing: {str(e)}")
        raise 