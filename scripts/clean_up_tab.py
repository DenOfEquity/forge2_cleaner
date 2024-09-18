import modules.scripts as scripts
import gradio as gr

from modules.shared import opts, OptionInfo
from modules import script_callbacks
from modules.ui_components import ToolButton, ResizeHandleRow
import modules.generation_parameters_copypaste as parameters_copypaste
from modules.ui_common import save_files

from scripts import lama

def on_ui_settings():
    section = ('cleaner', "Cleaner")
    opts.add_option("cleaner_use_gpu", OptionInfo(True, "Cleaner uses GPU", gr.Checkbox, {"interactive": True}, section=section))

def on_ui_tabs():
    with gr.Blocks() as object_cleaner_tab:
        
        for tab_name in ["Clean up", "Clean up upload"]:
            with gr.Tab(tab_name) as clean_up_tab, ResizeHandleRow():
                with gr.Column():
                    if tab_name == "Clean up":
                        init_img_with_mask = gr.ImageMask(
                            label="Image for clean up with mask", show_label=False,
                            elem_id="cleanup_img2maskimg", 
                            sources=['upload'], interactive=True, transforms=[''], layers=False,
                            type="pil", image_mode="RGBA", 
                            brush=gr.Brush(colors=['#FFFFFF'], color_mode='fixed'))
                    else:
                        with gr.Column(elem_id=f"cleanup_image_mask"):
                            clean_up_init_img = gr.Image(label="Image for cleanup", show_label=False, source="upload",
                                                         interactive=True, type="pil", elem_id="cleanup_img_inpaint_base", height=325)
                            clean_up_init_mask = gr.Image(
                                label="Mask", source="upload", interactive=True, type="pil", image_mode="RGBA", elem_id="cleanup_img_inpaint_mask", height=325)

                with gr.Column(elem_id=f"cleanup_gallery_container"):
                    clean_button = gr.Button("Clean Up")

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

                        send_to_cleaner_button = gr.Button("Send back To clean up")

            if tab_name == "Clean up":
                clean_button.click(
                    fn=lama.clean_object_init_img_with_mask,
                    inputs=[init_img_with_mask],
                    outputs=[result_gallery],
                    show_progress='full'
                )
                send_to_cleaner_button.click(
                    fn=lambda x: x,
                    inputs=[result_gallery],
                    outputs=[init_img_with_mask]
                )
            else:
                clean_button.click(
                    fn=lama.clean_object,
                    inputs=[clean_up_init_img, clean_up_init_mask],
                    outputs=[result_gallery],
                    show_progress='full'
                )
                send_to_cleaner_button.click(
                    fn=lambda x: x,
                    inputs=[result_gallery],
                    outputs=[clean_up_init_img]
                )

    return (object_cleaner_tab, "Cleaner", "cleaner_tab"),

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
