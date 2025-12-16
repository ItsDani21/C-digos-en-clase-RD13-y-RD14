
import os
# Set environment variables BEFORE cv2 usage to suppress logs
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet|timeout;20000|rtsp_transport;tcp'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

import cv2
import yt_dlp
from ultralytics import YOLO
import threading
import time
import numpy as np

class TrafficCamera:
    def __init__(self, youtube_url, model_path='yolov8n.pt'):
        self.youtube_url = youtube_url
        self.model = YOLO(model_path)
        self.video_url = None
        self.cap = None
        self.lock = threading.Lock()
        
        # Shared state for traffic counts
        self.traffic_state = {
            'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0,
            'frame': None
        }
        
        self.zones = {
            # Norte: Width reduced by half, Left Edge kept intact (0.0 -> 0.12, 0.25 -> 0.40)
            'norte': np.array([[0.00, 0.00], [0.12, 0.00], [0.40, 0.50], [0.25, 0.50]]),
            
            # Sur: Pushed down significantly (0.75) to make room
            'sur':   np.array([[0.35, 0.75], [0.90, 0.75], [0.90, 1.0], [0.30, 1.0]]),
            
            # Este: Moved down (Starts 0.40) to align with new pattern
            'este':  np.array([[0.65, 0.40], [1.0, 0.40], [1.0, 0.70], [0.65, 0.70]]),
            
            # Oeste: Moved down (Starts 0.55) to avoid Norte overlap
            'oeste': np.array([[0.0, 0.55], [0.25, 0.55], [0.25, 0.85], [0.0, 0.85]])
        }
        
        self.current_phase = "INIT" # Store current light phase

        self.running = False
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.daemon = True

    def set_phase(self, phase_id):
        # 0=NS Green, 1=EW Green (simplified mapping based on typical model output)
        if phase_id == 0:
            self.current_phase = "NS: GREEN | EW: RED"
        elif phase_id == 1:
            self.current_phase = "NS: RED | EW: GREEN"
        else:
            self.current_phase = f"PHASE: {phase_id}"

    def _get_stream_url(self):
        # Force HLS (m3u8) and IPv4 for stability
        ydl_opts = {
            'format': 'best[protocol^=m3u8]', 
            'quiet': True,
            'noplaylist': True,
            'force_ipv4': True 
        }
        self.video_url = None
        self.cap = None
        self.lock = threading.Lock()
        
        # Shared state for traffic counts
        self.traffic_state = {
            'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0,
            'frame': None
        }
        
        self.zones = {
            # Norte: Even wider to West (0.15) and deeper South (0.50)
            'norte': np.array([[0.15, 0.00], [0.65, 0.00], [0.60, 0.50], [0.25, 0.50]]),
            
            # Sur: Pushed down significantly (0.75) to make room
            'sur':   np.array([[0.35, 0.75], [0.90, 0.75], [0.90, 1.0], [0.30, 1.0]]),
            
            # Este: Moved down (Starts 0.40) to align with new pattern
            'este':  np.array([[0.65, 0.40], [1.0, 0.40], [1.0, 0.70], [0.65, 0.70]]),
            
            # Oeste: Moved down (Starts 0.55) to avoid Norte overlap
            'oeste': np.array([[0.0, 0.55], [0.25, 0.55], [0.25, 0.85], [0.0, 0.85]])
        }
        
        self.current_phase = "INIT" # Store current light phase

        self.running = False
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.daemon = True

    def set_phase(self, phase_id):
        # 0=NS Green, 1=EW Green (simplified mapping based on typical model output)
        if phase_id == 0:
            self.current_phase = "NS: GREEN | EW: RED"
        elif phase_id == 1:
            self.current_phase = "NS: RED | EW: GREEN"
        else:
            self.current_phase = f"PHASE: {phase_id}"

    def start(self):
        self.running = True
        self.thread.start()

    def _get_stream_url(self):
        # Force HLS (m3u8) which is much friendlier to OpenCV than DASH
        ydl_opts = {
            'format': 'best[protocol^=m3u8]', 
            'quiet': True,
            'noplaylist': True
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                # print(f"Format Found: {info.get('format_id')}")
                return info['url']
        except Exception as e:
            print(f"yt-dlp error: {e}")
            return None

    def _process_stream(self):
        print("Initializing Traffic Camera Stream...")
        
        # Limit FFMPEG verbosity and optimize for HLS stream stability
        # 'loglevel;quiet' hides errors, but 'timeout;20000' keeps connection alive longer
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet|timeout;20000|rtsp_transport;tcp'

        while self.running:
            # Reconnection Loop
            if self.cap is None or not self.cap.isOpened():
                stream_url = self._get_stream_url()
                if stream_url:
                    # Generic CAP_ANY or CAP_FFMPEG
                    self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                
                if not self.cap or not self.cap.isOpened():
                    print("Connection failed. Retrying in 5s...")
                    time.sleep(5)
                    continue
                else:
                    print("Stream Connected.")

            success, frame = self.cap.read()
            if not success:
                print("Read failed (Stream ended or lost). Reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(1) # Brief pause before retry
                continue

            # Resize frame
            frame = cv2.resize(frame, (854, 480)) # 480p is good balance
            h, w, _ = frame.shape

            # Run YOLO
            results = self.model(frame, verbose=False, classes=[2, 3, 5, 7]) # car, motorcycle, bus, truck

            counts = {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0}
            
            # Draw Zones
            for zone_name, poly in self.zones.items():
                pts = (poly * [w, h]).astype(np.int32)
                color = (0, 255, 255) # Yellow default
                # simple logic: highlight zone if it has active green light? 
                # (omitted for now to keep simple)
                cv2.polylines(frame, [pts], True, color, 1)
                
                # Label Zones
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00']) - 20
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(frame, zone_name.upper(), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Process Detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    
                    # Draw car center
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                    # Check which zone
                    for zone_name, poly in self.zones.items():
                        if cv2.pointPolygonTest((poly * [w, h]).astype(np.int32), (cx, cy), False) >= 0:
                            counts[zone_name] += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            break
            
            # Draw Counts on Frame
            cv2.putText(frame, f"N:{counts['norte']} S:{counts['sur']} E:{counts['este']} W:{counts['oeste']}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw AI Decision (Phase)
            cv2.putText(frame, f"AI DECISION: {self.current_phase}", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # Print to Terminal (User Request)
            print(f"[DETECTED] N:{counts['norte']} S:{counts['sur']} E:{counts['este']} W:{counts['oeste']} | Phase: {self.current_phase}")

            # Update State
            with self.lock:
                # User requested REAL detection counts (1 detected = 1 count)
                SCALE_FACTOR = 1 
                
                scaled_counts = {k: v * SCALE_FACTOR for k, v in counts.items()}
                self.traffic_state.update(scaled_counts)
                
                _, buffer = cv2.imencode('.jpg', frame)
                self.traffic_state['frame'] = buffer.tobytes()

            # Limit FPS to save CPU
            # time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.traffic_state['frame']

    def get_counts(self):
        with self.lock:
            return {
                'norte': self.traffic_state['norte'],
                'sur': self.traffic_state['sur'],
                'este': self.traffic_state['este'],
                'oeste': self.traffic_state['oeste']
            }
