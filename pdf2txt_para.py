import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


def process_pdf(pdf_file_path):
    try:
        # 获取文件名不含后缀
        name_without_suff = pdf_file_path.split(".")[0]
        print(f"Processing {pdf_file_path} as {name_without_suff}")
        
        # prepare env
        output_base_dir = Path("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/output")
        local_image_dir = output_base_dir / "images" / name_without_suff
        local_md_dir = output_base_dir / name_without_suff
        image_dir = str(os.path.basename(local_image_dir))
        
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)
        image_writer, md_writer = FileBasedDataWriter(str(local_image_dir)), FileBasedDataWriter(str(local_md_dir))
        image_dir = str(os.path.basename(local_image_dir))
        
        # read bytes
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_file_path)  # read the pdf content

        # proc
        ## Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        ## inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)

            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        ### draw model result on each page
        infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

        ### draw layout result on each page
        pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

        ### draw spans result on each page
        pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

        ### dump markdown
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

        ### dump content list
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)
        print(f"Finished processing {pdf_file_path}")
    except Exception as e:
        print(f"Error processing {pdf_file_path}: {e}")

def get_pdf_files(directory):
    """获取指定目录下所有 PDF 文件的路径"""
    return [str(p) for p in Path(directory).rglob("*.pdf")]

def main():
    # 指定包含 PDF 文件的文件夹路径
    pdf_folder = "/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data"

    # 获取所有 PDF 文件路径
    pdf_files = get_pdf_files(pdf_folder)

    if not pdf_files:
        print("没有找到 PDF 文件。")
        return

    print(f"找到 {len(pdf_files)} 个 PDF 文件。开始并行处理...")

    # 获取系统的 CPU 核心数量
    num_processes = cpu_count()

    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用 map 方法将文件列表分配给进程池
        pool.map(process_pdf, pdf_files)

    print("所有文件处理完毕。")

if __name__ == "__main__":
    main()
