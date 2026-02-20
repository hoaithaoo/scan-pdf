import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

def check_system():
    print("--- KIỂM TRA HỆ THỐNG ---")
    
    # 1. Kiểm tra Qdrant
    try:
        client = QdrantClient(
            url=f"http://{os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}",
            api_key=os.getenv('QDRANT_API_KEY')
        )
        collections = client.get_collections()
        print("✅ Kết nối Qdrant thành công!")
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")

    # 2. Kiểm tra Local Embedding (Sẽ tự tải model nếu lần đầu chạy)
    try:
        print("⏳ Đang kiểm tra Model Local (có thể mất chút thời gian lần đầu)...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        test_vec = model.encode(["Xin chào"])
        print(f"✅ Model Local hoạt động tốt. Vector size: {len(test_vec[0])}")
    except Exception as e:
        print(f"❌ Lỗi tải Model Local: {e}")

if __name__ == "__main__":
    check_system()