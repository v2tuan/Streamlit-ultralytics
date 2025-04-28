# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import queue
import threading
from typing import Any

import cv2
import numpy as np
import time

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the Inference class."""
        check_requirements("streamlit>=1.29.0")
        import streamlit as st

        self.st = st
        self.source = None
        self.enable_trk = "No"
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None

        # Using queues for thread-safe data passing
        self.frame_queue = queue.Queue(maxsize=1)
        self.annotated_frame_queue = queue.Queue(maxsize=1)
        self.running = False

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        # Create session state for controlling the app flow
        if 'running' not in self.st.session_state:
            self.st.session_state['running'] = False

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")
        self.source = self.st.sidebar.selectbox(
            "Video Source",
            ("webcam", "video"),
        )
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

    def source_upload(self):
        """Handle video file uploads through the Streamlit interface."""
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("ultralytics.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0

    def configure(self):
        """Configure the model and load selected classes for inference."""
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)
            
        return class_names

    def inference(self):
        """Perform real-time object detection inference."""
        self.web_ui()
        self.sidebar()
        self.source_upload()
        class_names = self.configure()
        
        # Create two columns for displaying frames
        col1 = self.st.columns(1)[0]
        
        with col1:
            self.st.write("Webcam Stream (Left: Original, Right: Processed)")
            self.org_frame = self.st.empty()
        
        if self.source == "video" and self.vid_file_name:
            if self.st.sidebar.button("Start"):
                stop_button = self.st.button("Stop")
                cap = cv2.VideoCapture(self.vid_file_name)
                if not cap.isOpened():
                    self.st.error("Could not open video source.")
                    return
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process with YOLO
                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                    
                    # Convert BGR to RGB for Streamlit display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Plot results on frame
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frames
                    self.org_frame.image(rgb_frame, channels="RGB")
                    self.ann_frame.image(annotated_frame_rgb, channels="RGB")
                    
                    # Check if stop button was pressed
                    if stop_button:
                        break
                
                cap.release()
        
        # Process webcam case with streamlit-webrtc
        elif self.source == "webcam":
            check_requirements("streamlit-webrtc>=0.45.0 av>=10.0.0")
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av
            
            class VideoProcessor(VideoProcessorBase):
                def __init__(self, model, conf, iou, selected_ind, enable_trk, class_names, frame_queue):
                    self.model = model
                    self.conf = conf
                    self.iou = iou
                    self.selected_ind = selected_ind
                    self.enable_trk = enable_trk
                    self.class_names = class_names
                    self.frame_queue = frame_queue
                    
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Put original frame in queue for display
                    try:
                        if self.frame_queue.full():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(img.copy())
                    except queue.Full:
                        pass
                    
                    # Process frame with model
                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                    
                    # Annotate the frame with detection results
                    annotated_frame = results[0].plot()
                    
                    # Create a combined frame with original and processed side by side
                    h, w = img.shape[:2]
                    combined_frame = np.zeros((h, w*2, 3), dtype=np.uint8)
                    combined_frame[:, :w] = img
                    combined_frame[:, w:] = annotated_frame
                    
                    return av.VideoFrame.from_ndarray(combined_frame, format="bgr24")
            
            # Configure WebRTC
            rtc_configuration = RTCConfiguration({
                "iceServers": [
                    {
                        "urls": "stun:global.stun.twilio.com:3478"
                    },
                    {
                        "urls": "turn:global.turn.twilio.com:3478?transport=udp",
                        "username": "e991673b3585edeabca62f206e6d3746a8db7dfb41533f892ea8960740528f2c",
                        "credential": "gD6NYi42L4wFD5Kt2C+Rx5sKiKYWY6UEfu1lzW6A9/w="
                    },
                    {
                        "urls": "turn:global.turn.twilio.com:3478?transport=tcp",
                        "username": "e991673b3585edeabca62f206e6d3746a8db7dfb41533f892ea8960740528f2c",
                        "credential": "gD6NYi42L4wFD5Kt2C+Rx5sKiKYWY6UEfu1lzW6A9/w="
                    },
                    {
                        "urls": "turn:global.turn.twilio.com:443?transport=tcp",
                        "username": "e991673b3585edeabca62f206e6d3746a8db7dfb41533f892ea8960740528f2c",
                        "credential": "gD6NYi42L4wFD5Kt2C+Rx5sKiKYWY6UEfu1lzW6A9/w="
                    }
                ]
            })
            
            # Function to update the original frame
            def update_original_frame():
                while True:
                    try:
                        if not self.frame_queue.empty():
                            frame = self.frame_queue.get()
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            self.org_frame.image(rgb_frame, channels="RGB")
                    except Exception as e:
                        self.st.error(f"Error updating original frame: {e}")
                    time.sleep(0.03)  # Update every 30ms (approx. 30fps)
            
            # Start WebRTC streamer in the second column
            with col1:
                webrtc_ctx = webrtc_streamer(
                    key="ultralytics-detection",
                    video_processor_factory=lambda: VideoProcessor(
                        self.model, self.conf, self.iou, self.selected_ind, self.enable_trk, class_names, self.frame_queue
                    ),
                    rtc_configuration=rtc_configuration,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                
                # Start the update thread when WebRTC is active
                if webrtc_ctx.state.playing:
                    if not hasattr(self, "_webrtc_thread") or not self._webrtc_thread.is_alive():
                        self._webrtc_thread = threading.Thread(target=update_original_frame, daemon=True)
                        self._webrtc_thread.start()
            
            # Information about WebRTC
            self.st.info("""
            ** Lưu ý về Webcam:**
            - Thông qua WebRTC, ứng dụng sẽ cần quyền truy cập vào webcam của bạn
            - Xử lý video được thực hiện trong trình duyệt của bạn
            - Hãy chắc chắn bạn đã cấp quyền truy cập webcam khi được hỏi
            """)


if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()
