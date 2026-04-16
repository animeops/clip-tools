import gradio as gr
from cliptools import FileProcessor

def process_files(paths):
    output_paths = []
    for path in paths:
        processor = FileProcessor(path)
        folders = processor.export(ext="png")
        output_paths.append(folders[0])
    return output_paths

demo = gr.Interface(
    fn=process_files, 
    inputs=gr.File(
        label="ファイル（.psdまたは.clip）をアップロードしてください", 
        file_types=[".clip", ".psd"], 
        file_count='multiple'
    ),
    outputs="file",
    title="書き出し君"
)
demo.launch()
