import gradio as gr
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering,  VisionEncoderDecoderModel
)
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import torch


# Load models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")




def generate_caption(image):
    if image is None:
        return "Please upload an image."
    img = Image.fromarray(image).convert("RGB")
    inputs = caption_processor(img, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def answer_question(image, question):
    if image is None:
        return "Please upload an image."
    img = Image.fromarray(image).convert("RGB")
    inputs = vqa_processor(img, question, return_tensors="pt")
    out = vqa_model.generate(**inputs)
    return vqa_processor.decode(out[0], skip_special_tokens=True)

def apply_filter_control(image, blur_level=0, grayscale_level=0, sharpen_level=1,contrast_level=1):
    if image is None:
        return None

    img = Image.fromarray(image).convert("RGB")
    if blur_level > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_level))
    if grayscale_level > 0:
        gray = ImageOps.grayscale(img).convert("RGB")
        img = Image.blend(img, gray, grayscale_level)
    
    if sharpen_level != 1:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpen_level)

    if contrast_level != 1:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_level)
    
    
    return img


with gr.Blocks(title="🌟Vision Language AI Multi-Function App") as demo:
    gr.Markdown("# Vision Language AI App\nUpload an image and explore multiple AI functions!")

    with gr.Row():
        img = gr.Image(type="numpy", label="Upload Image")

    with gr.Tabs():
        # Caption Tab
        with gr.Tab("📝Caption"):
            cap_btn = gr.Button("Generate Caption")
            cap_output = gr.Textbox(label="Caption Result")
            cap_btn.click(fn=generate_caption, inputs=img, outputs=cap_output)

        # VQA Tab
        with gr.Tab("❓VQA (Ask Image)"):
            q = gr.Textbox(label="Question about image")
            vqa_btn = gr.Button("Ask")
            vqa_output = gr.Textbox(label="Answer")
            vqa_btn.click(fn=answer_question, inputs=[img, q], outputs=vqa_output)

        # Image Filters Tab (可調整)
        with gr.Tab("🎨Image Filters"):
            blur_slider = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Blur Level")
            gray_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label="Grayscale Level")
            sharpen_slider = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Sharpen Level")
            contrast_slider = gr.Slider(minimum=0.1, maximum=3, step=0.01, value=1, label="Contrast Level")
            filter_btn = gr.Button("Apply Filter")
            filter_output = gr.Image(label="Filtered Image")
            filter_btn.click(
                fn=apply_filter_control,
                inputs=[img, blur_slider, gray_slider, sharpen_slider,contrast_slider],
                outputs=filter_output
            )

demo.launch()