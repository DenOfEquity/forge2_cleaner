import modules.scripts as scripts
import gradio as gr

from modules.shared import opts, OptionInfo
from modules import script_callbacks
from modules.ui_components import ToolButton
import modules.generation_parameters_copypaste as parameters_copypaste
from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
from importlib import reload

from scripts import lama

def on_ui_settings():
    section = ('cleaner', "Cleaner")
    opts.add_option("cleaner_use_gpu", OptionInfo(True, "Cleaner uses GPU", gr.Checkbox, {"interactive": True}, section=section))

def on_ui_tabs():
    reload(lama)

    with gr.Blocks(analytics_enabled=False, head=canvas_head) as cleaner_block:

        with gr.Row():
            with gr.Column():
                with gr.Tab("Clean up", id="cleaner_clean_up"):
                    cleaner_canvas = ForgeCanvas(elem_id="Cleaner_image", height=512, scribble_color=opts.img2img_inpaint_mask_brush_color, scribble_color_fixed=True, scribble_alpha=75, scribble_alpha_fixed=True, scribble_softness_fixed=True)

                    clean_up = gr.Button("Clean Up")
                    send_to_clean_up = gr.Button("fetch result")

                with gr.Tab("Upload", id="cleaner_upload"):
                    clean_up_init_img = gr.Image(show_label=False, source="upload", interactive=True, type="pil", height=325, elem_id="cleanup_img_inpaint_base")
                    clean_up_init_mask = gr.Image(label="Mask", source="upload", interactive=True, type="pil", image_mode="RGBA", height=325, elem_id="cleanup_img_inpaint_mask")
        
                    clean_upload = gr.Button("Clean Up")
                    send_to_clean_upload = gr.Button("fetch result")


            with gr.Column(elem_id=f"cleanup_gallery_container"):
                result_gallery = gr.Image(label='Output', height="60vh", type='pil', interactive=False, elem_id="cleanup_gallery", show_label=False, visible=True)

                with gr.Row(elem_id=f"image_buttons", elem_classes="image-buttons"):
                    buttons = {
                        'img2img': ToolButton('🖼️', elem_id=f'_send_to_img2img', tooltip="Send image and generation parameters to img2img tab."),
                        'inpaint': ToolButton('🎨️', elem_id=f'_send_to_inpaint', tooltip="Send image and generation parameters to img2img inpaint tab."),
                        'extras': ToolButton('📐', elem_id=f'_send_to_extras', tooltip="Send image and generation parameters to extras tab.")
                    }

                    for paste_tabname, paste_button in buttons.items():
                        parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                            paste_button=paste_button, tabname=paste_tabname, source_tabname=None, source_image_component=result_gallery,
                            paste_field_names=[]
                        ))

        clean_up.click(
            fn=lama.clean_object,
            inputs=[cleaner_canvas.background, cleaner_canvas.foreground],
            outputs=[result_gallery],
            show_progress='full'
        )
        send_to_clean_up.click(
            fn=lambda x: x,
            inputs=[result_gallery],
            outputs=[cleaner_canvas.background]
        )
        clean_upload.click(
            fn=lama.clean_object,
            inputs=[clean_up_init_img, clean_up_init_mask],
            outputs=[result_gallery],
            show_progress='full'
        )
        send_to_clean_upload.click(
            fn=lambda x: x,
            inputs=[result_gallery],
            outputs=[clean_up_init_img]
        )

    return (cleaner_block, "Cleaner", "cleaner_tab"),

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
