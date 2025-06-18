from flask import Flask, render_template, request, send_file, jsonify
import os
import numpy as np
from PIL import Image
import cv2
from rembg import remove, new_session
import uuid
from datetime import datetime

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
AVATAR_FOLDER = 'static/avatar'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AVATAR_FOLDER, exist_ok=True)

# Global session for rembg to avoid re-loading model
session = new_session("u2net_human_seg")

def remove_background_and_smooth(clothing_path):
    """
    Remove background from clothing image using rembg with u2net_human_seg model.
    Apply alpha matting and smooth the edges.
    """
    try:
        # Load the clothing image
        clothing_img = Image.open(clothing_path).convert("RGBA")
        
        # Remove background using rembg with alpha matting
        segmented_img = remove(
            clothing_img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        
        # Convert to numpy array for processing
        img_array = np.array(segmented_img)
        
        # Ensure background is fully transparent (set very low alpha values to 0)
        alpha = img_array[:, :, 3]
        alpha[alpha < 10] = 0
        
        # Apply slight Gaussian blur to alpha channel for smoother edges
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        # Reconstruct the image with the processed alpha channel
        img_array[:, :, 3] = alpha
        processed_img = Image.fromarray(img_array)
        
        return processed_img
        
    except Exception as e:
        print(f"Error in background removal: {str(e)}")
        return None

def crop_to_bounding_box(processed_img):
    """
    Crop the processed clothing image to its bounding box.
    """
    try:
        # Find all non-transparent pixels to determine the bounding box
        coords = np.argwhere(np.array(processed_img)[:, :, 3] > 0)
        
        if len(coords) == 0:
            print("No clothing pixels detected after segmentation. Cannot crop.")
            return None
        
        # Get bounding box coordinates (min_row, min_col, max_row, max_col)
        y_min, x_min = coords.min(axis=0)[:2]
        y_max, x_max = coords.max(axis=0)[:2]
        
        # Add some padding around the bounding box
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(processed_img.width, x_max + padding)
        y_max = min(processed_img.height, y_max + padding)
        
        # Crop the image to the bounding box
        cropped_img = processed_img.crop((x_min, y_min, x_max, y_max))
        
        return cropped_img
        
    except Exception as e:
        print(f"Error in bounding box cropping: {str(e)}")
        return None

def overlay_clothing_on_avatar(avatar_path, cropped_clothing, output_path):
    """
    Intelligently resize and position the clothing on the avatar.
    """
    try:
        # Load the fixed avatar
        avatar = Image.open(avatar_path).convert("RGBA")
        avatar_width, avatar_height = avatar.size
        
        print(f"Avatar size: {avatar.size}")
        print(f"Cropped clothing size: {cropped_clothing.size}")
        
        # Calculate target clothing size (adjust these ratios as needed)
        # Clothing should be about 50% of avatar width for natural fit
        target_clothing_width = int(avatar_width * 0.5)
        
        # Calculate height maintaining aspect ratio
        original_width, original_height = cropped_clothing.size
        if original_width == 0:
            print("Error: Cropped clothing width is zero.")
            return None
            
        aspect_ratio = original_height / original_width
        target_clothing_height = int(target_clothing_width * aspect_ratio)
        
        # Resize clothing to target size
        resized_clothing = cropped_clothing.resize(
            (target_clothing_width, target_clothing_height), 
            Image.LANCZOS
        )
        
        print(f"Resized clothing size: {resized_clothing.size}")
        
        # Calculate position to center the clothing on the avatar's upper body
        x_offset = (avatar_width - target_clothing_width) // 2
        # Position vertically (adjust this to fit on upper body/torso)
        y_offset = int(avatar_height * 0.2)  # 20% down from top
        
        # Create a new image with the same size as avatar
        composite = Image.new("RGBA", avatar.size)
        
        # Paste the avatar as background
        composite.paste(avatar, (0, 0))
        
        # Paste the clothing onto the avatar using alpha channel as mask
        composite.paste(resized_clothing, (x_offset, y_offset), resized_clothing)
        
        # Save the composite image
        composite.save(output_path, "PNG")
        
        print(f"Composite image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error in overlaying clothing: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_process():
    """Handle clothing image upload and processing."""
    try:
        # Check if clothing image was uploaded
        if 'clothing' not in request.files:
            return jsonify({'error': 'No clothing image uploaded'}), 400
        
        clothing_file = request.files['clothing']
        
        if clothing_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique filename for uploaded clothing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        clothing_filename = f"clothing_{timestamp}_{unique_id}.png"
        clothing_path = os.path.join(UPLOAD_FOLDER, clothing_filename)
        
        # Save uploaded clothing image
        clothing_file.save(clothing_path)
        print(f"Clothing image saved to: {clothing_path}")
        
        # Step 1: Remove background and smooth edges
        print("Step 1: Removing background and smoothing edges...")
        processed_clothing = remove_background_and_smooth(clothing_path)
        
        if processed_clothing is None:
            return jsonify({'error': 'Failed to remove background from clothing image'}), 500
        
        # Step 2: Crop to bounding box
        print("Step 2: Cropping to bounding box...")
        cropped_clothing = crop_to_bounding_box(processed_clothing)
        
        if cropped_clothing is None:
            return jsonify({'error': 'Failed to crop clothing image'}), 500
        
        # Step 3: Overlay on fixed avatar
        print("Step 3: Overlaying clothing on avatar...")
        
        # Use a fixed avatar (you can replace this with your preferred avatar)
        avatar_path = os.path.join(AVATAR_FOLDER, 'default_avatar.png')
        
        # If no default avatar exists, create a simple placeholder
        if not os.path.exists(avatar_path):
            # Create a simple placeholder avatar (you can replace this with your actual avatar)
            placeholder_avatar = Image.new('RGBA', (400, 600), (240, 240, 240, 255))
            # Add a simple face shape
            from PIL import ImageDraw
            draw = ImageDraw.Draw(placeholder_avatar)
            # Draw a simple head
            draw.ellipse([150, 50, 250, 150], fill=(255, 218, 185, 255), outline=(0, 0, 0, 255))
            # Draw a simple body
            draw.rectangle([175, 150, 225, 400], fill=(100, 149, 237, 255), outline=(0, 0, 0, 255))
            placeholder_avatar.save(avatar_path)
        
        # Generate output filename
        output_filename = f"composite_{timestamp}_{unique_id}.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Overlay clothing on avatar
        success = overlay_clothing_on_avatar(avatar_path, cropped_clothing, output_path)
        
        if not success:
            return jsonify({'error': 'Failed to create composite image'}), 500
        
        # Clean up uploaded file
        os.remove(clothing_path)
        
        # Return the result
        return jsonify({
            'success': True,
            'result_image': f'/static/output/{output_filename}',
            'message': 'Clothing successfully overlaid on avatar!'
        })
        
    except Exception as e:
        print(f"Error in upload_and_process: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/static/output/<filename>')
def serve_output(filename):
    """Serve the generated composite images."""
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    print("Starting AIstylist Flask app...")
    print("Upload folder:", UPLOAD_FOLDER)
    print("Output folder:", OUTPUT_FOLDER)
    print("Avatar folder:", AVATAR_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000) 