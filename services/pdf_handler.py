import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import os

# Cài đặt log để theo dõi tiến trình
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFHandler:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # chunk_size: Độ dài mỗi đoạn (khoảng 800 ký tự)
        # chunk_overlap: Số ký tự gối đầu nhau giữa 2 đoạn liên tiếp để không mất ngữ cảnh
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_pdf(self, file_path: str):
        documents = []
        logger.info(f"Đang tiến hành đọc và bóc tách file: {file_path}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Duyệt qua từng trang của file PDF
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Cắt văn bản của trang hiện tại thành các đoạn nhỏ
                    chunks = self.splitter.split_text(text)
                    
                    # Lấy tên file gốc (bỏ đi các đường dẫn thư mục rườm rà)
                    file_name = os.path.basename(file_path)
                    
                    for chunk in chunks:
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": file_name,
                                "page": i + 1  # Đánh dấu số trang
                            }
                        })
            
            logger.info(f"✅ Đã cắt file thành công: Tổng cộng {len(documents)} đoạn văn.")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình xử lý PDF: {e}")
            raise e