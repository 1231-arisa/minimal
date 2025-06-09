import gradio as gr
import os
import numpy as np
from PIL import Image
import cv2
from rembg import remove, new_session

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

def process_images(avatar_image, clothing_image):
    if avatar_image is None or clothing_image is None:
        return None, "Please upload both avatar and clothing images."
    try:
        avatar = Image.fromarray(avatar_image)
        clothing = Image.fromarray(clothing_image)
        avatar_path = "input/temp_avatar.png"
        clothing_path = "input/temp_clothing.png"
        output_path = "output/result.png"
        avatar.save(avatar_path)
        clothing.save(clothing_path)
        processed_clothing = segment_and_crop_clothing(clothing_path)
        if processed_clothing is None:
            return None, "Clothing segmentation failed."
        composite_images(avatar_path, processed_clothing, output_path)
        result = Image.open(output_path)
        return np.array(result), "Successfully composited!"
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_interface():
    with gr.Blocks(title="AIstylist - Clothing Avatar Composer") as interface:
        gr.Markdown("""
        # 🧠 AIstylist - Clothing Avatar Composer
        Upload an avatar image and a clothing image to create a composite.
        """)
        with gr.Row():
            with gr.Column():
                avatar_input = gr.Image(label="Avatar Image", type="numpy")
                clothing_input = gr.Image(label="Clothing Image", type="numpy")
                process_btn = gr.Button("Process Images")
            with gr.Column():
                output_image = gr.Image(label="Composite Result")
                output_text = gr.Textbox(label="Status")
        process_btn.click(
            fn=process_images,
            inputs=[avatar_input, clothing_input],
            outputs=[output_image, output_text]
        )
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
