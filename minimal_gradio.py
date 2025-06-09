import gradio as gr
import os
from PIL import Image
import numpy as np
from main_clothing_avatar_composite import segment_and_crop_clothing, composite_images

def process_images(avatar_image, clothing_image):
    if avatar_image is None or clothing_image is None:
        return None, "Please upload both avatar and clothing images."

    try:
        # Convert numpy arrays to PIL Images
        avatar = Image.fromarray(avatar_image)
        clothing = Image.fromarray(clothing_image)

        # Save temp files
        avatar_path = "input/temp_avatar.png"
        clothing_path = "input/temp_clothing.png"
        output_path = "output/result.png"
        avatar.save(avatar_path)
        clothing.save(clothing_path)

        # Process clothing
        processed_clothing = segment_and_crop_clothing(clothing_path)
        if processed_clothing is None:
            return None, "Clothing segmentation failed."

        # Composite
        composite_images(avatar_path, processed_clothing, output_path)

        # Return result image
        result = Image.open(output_path)
        return np.array(result), "Successfully composited!"

    except Exception as e:
        return None, f"Error: {str(e)}"

# Create the Gradio interface
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
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Makes the server accessible from other devices
        server_port=7860,  # Default Gradio port
        share=False  # Disable share link to avoid connection issues
    )
