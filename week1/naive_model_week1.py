from langgraph.graph import StateGraph, END
from typing import TypedDict
from backend import editor, loader
import reader 
import random
import subprocess
import os
import uuid

# Định nghĩa state dùng chung
class AgentState(TypedDict):
    video_path: str
    domain: str           # "lecture" | "podcast" | "standup"
    source_video: str
    workspace_dir: str
    profile: dict         # trọng số từng tầng theo miền
    features: dict        # kết quả 5 tầng trích xuất
    candidates: list      # danh sách ứng viên đã chấm điểm
    highlights: list      # top-K đã chọn + boundary
    reasoning: list       # văn bản giải thích từng highlight
    
class InputState(TypedDict): 
    video_path: str 
    domain: str
    
class PlanState(TypedDict):
    domain: dict
    

def extract_video_id(video_input: str) -> str:
    """
    Tự động trích xuất ID từ bất kỳ link YouTube hoặc file cục bộ nào.
    Không bao giờ bị fix cứng vào một video cụ thể.
    """
    if not video_input or str(video_input) == "None":
        # Nếu không có đầu vào, tạo một ID ngẫu nhiên để không bị lỗi
        return f"video_{uuid.uuid4().hex[:8]}"
        
    video_str = str(video_input)
    
    # 1. Trường hợp link YouTube chuẩn (vd: https://www.youtube.com/watch?v=jbL9kl4KPZI)
    if "v=" in video_str:
        return video_str.split("v=")[-1].split("&")[0]
        
    # 2. Trường hợp link YouTube rút gọn (vd: https://youtu.be/jbL9kl4KPZI)
    elif "youtu.be/" in video_str:
        return video_str.split("youtu.be/")[-1].split("?")[0]
        
    # 3. Trường hợp truyền đường dẫn file trên máy (vd: input/my_lecture.mp4 -> lấy 'my_lecture')
    elif "/" in video_str or "\\" in video_str:
        return os.path.splitext(os.path.basename(video_str))[0]
        
    # 4. Trường hợp khác (đã là ID sẵn)
    return video_str

# Mỗi pha là một hàm Python — LangGraph tự gọi theo thứ tự
def observe(state: InputState):
    video_input = state.get('video_path')
    workspace_id = reader.get_workspace_id(video_input)
    workspace_dir = f'processing_video/{workspace_id}'
    os.makedirs(workspace_dir, exist_ok = True)
    
    audio_path = f'{workspace_dir}/audio.wav'
    transcript_path = f'{workspace_dir}/transcript.json'
    
    print(f'ID workspace: {workspace_dir}')
    
    media_info = loader.prepare_media_workspace(video_input, workspace_dir)
    source_video_local = media_info.get("source_video", video_input)
    
    dectected_domain = state.get('domain')
    return {
        "source_video": source_video_local,
        "workspace_dir": workspace_dir,   
        "transcript": transcript_path
    }

def plan(state: AgentState) -> dict: 
    domain = state.get('domain')
    print(f'Mien noi dung cua video hien tai: f{domain}')
    
    if domain == 'lecture':
        profile_weight = {
            "aucostic": 67,
            "paralinguistic": 67, 
            "linguistic": 36, 
            "structural": 18, 
            "interactive": 67
        }
        
    elif domain == 'podcast': 
        profile_weight = {
            "aucostic": 67,
            "paralinguistic": 67, 
            "linguistic": 36, 
            "structural": 18, 
            "interactive": 67
        }
    elif domain == 'standup': 
        profile_weight = {
            "aucostic": 67,
            "paralinguistic": 67, 
            "linguistic": 36, 
            "structural": 18, 
            "interactive": 67
        }
    else: #fallback neu khong xac dinh duoc mien
        profile_weight = {
            "aucostic": 67,
            "paralinguistic": 67, 
            "linguistic": 36, 
            "structural": 18, 
            "interactive": 67
        }
        
    print('Xac dinh profile_weight: ', profile_weight)
    return {'profile': profile_weight}



def analyze(state: AgentState) -> dict: 
    video_path = state.get('video_path')
    profile = state.get('profile')
    
    #tuan 2 se fill phan trich xuat dac trung vao day 
    
    dummy_features = {
        'acoustic': 'none', 
        'linguistic': 'none'
    }
    
    dummy_candidates = []
    
    for i in range(5): 
        start_time = random.randint(0, 300)
        end_time = start_time + 60
        
        mock_score = round(random.uniform(2.0, 5.0), 2)
        
        dummy_candidates.append({
            'candidate_id': f'cand_{i + 1}', 
            'start_time': start_time,
            'end_time': end_time,
            'score': mock_score,
            'reason': 'random'
        })
        
        dummy_candidates.sort(key = lambda x : x['score'], reverse = True)
        
    print('chon xong')
    
    return {'features': dummy_features, 
            "candidates": dummy_candidates}
    
def decide(state: AgentState) -> dict: 
    print('loc candidate')
    
    #lay du lieu tu node truoc
    candidates = state.get('candidates', [])
    source_video = state.get('source_video')
    workspace_dir = state.get('workspace_dir')
    
    if not workspace_dir:
        video_input = state.get('video_path') 
        video_id = extract_video_id(video_input)
        workspace_dir = os.path.join('processing_video', video_id)
    
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Nếu source_video là URL hoặc không tồn tại, kiểm tra file local trong workspace_dir
    local_source_video = os.path.join(workspace_dir, "source_video.mp4")
    if os.path.exists(local_source_video):
        source_video = local_source_video

    #sap xep candidate theo diem so tu cao xuong thap
    sorted_candidates = sorted(candidates, key = lambda x : x['score'], reverse = True)
    
    top_3_hl = sorted_candidates[:3]
    
    short_dir = os.path.join(workspace_dir, 'shorts')
    os.makedirs(short_dir, exist_ok=True)
    
    output_files = []
    
    for index, hl in enumerate(top_3_hl):
        start = hl['start_time']
        end = hl['end_time']
        
        file_name = f'highlight_{index + 1}.mp4'
        out_path = os.path.join(short_dir, file_name)
        
        print(f" -> Đang cắt clip #{index+1} (Từ giây {start} đến {end})...")
        
        # Lệnh FFmpeg cắt clip + crop dọc 9:16
        command = [
            "ffmpeg", "-y",              # -y: Tự động ghi đè nếu file đã tồn tại
            "-ss", str(start),           # Giây bắt đầu
            "-to", str(end),             # Giây kết thúc
            "-i", source_video,          # Video gốc
            "-vf", "crop=ih*(9/16):ih",  # Cắt khung hình ngang thành dọc 9:16 (lấy chính giữa)
            "-c:v", "libx264",           # Chuẩn nén video phổ biến
            "-preset", "ultrafast",      # ultrafast: Cắt siêu nhanh cho Tuần 1 demo
            out_path                     # Nơi lưu file thành phẩm
        ]
        
        # Chạy lệnh FFmpeg (ẩn các dòng log rác của ffmpeg cho sạch terminal)
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Lưu đường dẫn file vừa cắt xong vào danh sách
        output_files.append(out_path)
        
    print(f" -> Đã cắt xong {len(output_files)} clip Shorts!")
    
    # 6. Trả về kết quả để gửi sang Node tiếp theo
    return {
        "highlights": top_3_hl,
        "output_files": output_files
    } 
    
    
def explain(state: AgentState) -> dict: 
    highlights = state.get('highlights', [])
    domain = state.get('domain', 'lecture')
    
    reasoning_list = []
    
    for index, hl in enumerate(highlights):
        start = hl['start_time']
        end = hl['end_time']
        score = hl['score']
        cand_id = hl['candidate_id']
        
        cau_giai_thich = (
            f"Clip #{index+1} [{start}s - {end}s] | Điểm đánh giá: {score}/5.0\n"
            f"• Lý do chọn: Đây là đoạn clip có điểm số cao trong danh sách kiểm thử Tuần 1 "
            f"đối với video thuộc miền nội dung '{domain.upper()}'.\n"
            f"• Khung hình: Hệ thống đã cắt tự động khu vực trung tâm sang tỷ lệ dọc 9:16."
        )
        
        reasoning_list.append({
            'candidate_id':cand_id, 
            'explanation': cau_giai_thich
        })
        
    return {"reasoning" : reasoning_list}
# Dựng đồ thị
graph = StateGraph(AgentState)
graph.add_node("observe", observe)
graph.add_node("plan", plan)
graph.add_node("analyze", analyze)                                                                                                                                                                                                      
graph.add_node("decide", decide)
graph.add_node("explain", explain)

# Nối các node theo thứ tự
graph.set_entry_point("observe")
graph.add_edge("observe", "plan")
graph.add_edge("plan", "analyze")
graph.add_edge("analyze", "decide")
graph.add_edge("decide", "explain")
graph.add_edge("explain", END)

app = graph.compile()
app.invoke({"video_path": "https://www.youtube.com/watch?v=jbL9kl4KPZI", 'domain' : 'lecture'})
# Stream: for step in app.stream(state): print(step

