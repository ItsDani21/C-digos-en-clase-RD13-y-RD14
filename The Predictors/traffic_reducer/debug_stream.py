
import cv2
import yt_dlp
import time

VIDEO_URL = "https://www.youtube.com/watch?v=ByED80IKdIU"

def get_stream_url(url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info['url']
        except Exception as e:
            print(f"Error extracting URL: {e}")
            return None

print("Extracting URL...")
stream_url = get_stream_url(VIDEO_URL)
print(f"URL Found: {stream_url[:50]}...")

if stream_url:
    print("Attempting to open with OpenCV...")
    # CAP_FFMPEG needed for streams usually
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("FAILED to open video capture")
    else:
        print("SUCCESS! Video capture opened.")
        ret, frame = cap.read()
        if ret:
            print(f"Frame read success. Shape: {frame.shape}")
        else:
            print("Failed to read first frame")
        cap.release()
else:
    print("Could not get stream URL")
