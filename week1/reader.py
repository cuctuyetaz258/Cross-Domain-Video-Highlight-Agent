import os
import re
import hashlib

def get_workspace_id(video_input: str) -> str:
    """_summary_
    Hàm chuẩn hoá đầu vào thành ID để đặt tên thư mục
    """
    #Input là link youtube
    if 'youtube.com' in video_input or 'youtu.be' in video_input: 
        match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', video_input)
        if match: 
            return match.group(1)
        else:
            return hashlib.md5(video_input.encode().hexdigest())[:10]
    
    else:
        clean_name = os.path.basename(video_input)
        return os.path.splitext(clean_name)[0]