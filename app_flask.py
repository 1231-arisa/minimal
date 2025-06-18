from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
from PIL import Image
import cv2
from rembg import remove, new_session
import base64
import io
import uuid

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def segment_and_crop_clothing(clothing_path):
    """
    Segments the clothing from its background, applies alpha matting,
    smooths edges, and crops the image to the bounding box of the clothing.
    """
    session = new_session("u2net_human_seg")
    try:
        original_clothing_img = Image.open(clothing_path).convert("RGBA")
        segmented_img = remove(
            original_clothing_img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        img_array = np.array(segmented_img)
        alpha = img_array[:, :, 3]
        alpha[alpha < 10] = 0
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        img_array[:, :, 3] = alpha
        processed_segmented_img = Image.fromarray(img_array)
        coords = np.argwhere(np.array(processed_segmented_img)[:, :, 3] > 0)
        if len(coords) == 0:
            return None
        y_min, x_min = coords.min(axis=0)[:2]
        y_max, x_max = coords.max(axis=0)[:2]
        cropped_clothing = processed_segmented_img.crop((x_min, y_min, x_max, y_max))
        return cropped_clothing
    except Exception as e:
        return None

def composite_images(avatar_path, processed_clothing_img, output_path):
    try:
        avatar = Image.open(avatar_path).convert("RGBA")
        if processed_clothing_img is None:
            return
        avatar_width, avatar_height = avatar.size
        target_clothing_width_on_avatar = int(avatar_width * 0.5)
        original_cropped_width, original_cropped_height = processed_clothing_img.size
        if original_cropped_width == 0:
            return
        aspect_ratio = original_cropped_height / original_cropped_width
        target_clothing_height_on_avatar = int(target_clothing_width_on_avatar * aspect_ratio)
        if target_clothing_width_on_avatar == 0 or target_clothing_height_on_avatar == 0:
            return
        clothing_to_paste = processed_clothing_img.resize(
            (target_clothing_width_on_avatar, target_clothing_height_on_avatar), Image.LANCZOS
        )
        x_offset = (avatar_width - clothing_to_paste.size[0]) // 2
        y_offset = int(avatar_height * 0.18)
        composite = Image.new("RGBA", avatar.size)
        composite.paste(avatar, (0, 0))
        composite.paste(clothing_to_paste, (x_offset, y_offset), clothing_to_paste)
        composite.save(output_path, "PNG")
    except Exception as e:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    try:
        # Get uploaded files
        avatar_file = request.files['avatar']
        clothing_file = request.files['clothing']
        
        if not avatar_file or not clothing_file:
            return jsonify({'error': 'Please upload both avatar and clothing images'}), 400
        
        # Generate unique filenames
        avatar_filename = f"avatar_{uuid.uuid4().hex}.png"
        clothing_filename = f"clothing_{uuid.uuid4().hex}.png"
        output_filename = f"result_{uuid.uuid4().hex}.png"
        
        avatar_path = os.path.join(UPLOAD_FOLDER, avatar_filename)
        clothing_path = os.path.join(UPLOAD_FOLDER, clothing_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Save uploaded files
        avatar_file.save(avatar_path)
        clothing_file.save(clothing_path)
        
        # Process images
        processed_clothing = segment_and_crop_clothing(clothing_path)
        if processed_clothing is None:
            return jsonify({'error': 'Clothing segmentation failed'}), 500
        
        composite_images(avatar_path, processed_clothing, output_path)
        
        # Convert result to base64 for sending back
        with open(output_path, 'rb') as f:
            result_data = f.read()
        result_base64 = base64.b64encode(result_data).decode('utf-8')
        
        # Clean up temporary files
        os.remove(avatar_path)
        os.remove(clothing_path)
        os.remove(output_path)
        
        return jsonify({
            'success': True,
            'result': result_base64,
            'message': 'Successfully composited!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860) 