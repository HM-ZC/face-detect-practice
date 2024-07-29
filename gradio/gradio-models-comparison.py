import cv2
import dlib
import imutils
import numpy as np
import mediapipe as mp
import torch
import torchlm
import gradio as gr
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from torchlm.runtime import faceboxesv2_ort, pipnet_ort

class FaceModelsComparison:
    def __init__(self):
        self.detector_dlib = dlib.get_frontal_face_detector()
        self.predictor_dlib = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = "../model/pipnet_resnet101_10x68x32x256_300w.pth"

        torchlm.runtime.bind(faceboxesv2())
        torchlm.runtime.bind(
            pipnet(
                backbone="resnet101",
                pretrained=True,
                num_nb=10,
                num_lms=68,
                net_stride=32,
                input_size=256,
                meanface_type="300w",
                backbone_pretrained=False,
                map_location=self.device,
                checkpoint=self.checkpoint
            )
        )

    def draw_landmarks(self, image, landmarks, color):
        for (x, y) in landmarks:
            cv2.circle(image, (int(x), int(y)), 2, color, -1)
        return image

    def draw_lines_between_models(self, image, landmarks1, landmarks2):
        if len(landmarks1) == 0 or len(landmarks2) == 0:
            return image

        # Convert landmarks to numpy arrays for distance calculation
        landmarks1 = np.array(landmarks1)
        landmarks2 = np.array(landmarks2)

        for pt1 in landmarks1:
            distances = np.linalg.norm(landmarks2 - pt1, axis=1)
            index = np.argmin(distances)
            pt2 = landmarks2[index]
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        return image

    def process_frame(self, frame):
        # Dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector_dlib(gray, 0)
        dlib_landmarks = []

        for rect in rects:
            shape = self.predictor_dlib(gray, rect)
            dlib_landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # MediaPipe
        mp_landmarks = []
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]

        # Torchlm
        torchlm_landmarks, _ = torchlm.runtime.forward(frame)
        if torchlm_landmarks.size > 0:
            torchlm_landmarks = torchlm_landmarks[0]
        else:
            torchlm_landmarks = []

        return dlib_landmarks, mp_landmarks, torchlm_landmarks

    def create_orientation(self, frame):
        frame = imutils.resize(frame, width=640)
        dlib_landmarks, mp_landmarks, torchlm_landmarks = self.process_frame(frame)

        # Original frame
        original_frame = frame.copy()

        # MediaPipe landmarks
        mp_frame = frame.copy()
        self.draw_landmarks(mp_frame, mp_landmarks, (255, 0, 0))

        # Torchlm landmarks
        torchlm_frame = frame.copy()
        self.draw_landmarks(torchlm_frame, torchlm_landmarks, (0, 0, 255))

        # Comparison frame with lines between corresponding landmarks
        comparison_frame = frame.copy()
        self.draw_landmarks(comparison_frame, mp_landmarks, (255, 0, 0))
        self.draw_landmarks(comparison_frame, torchlm_landmarks, (0, 0, 255))
        if len(mp_landmarks) > 0 and len(torchlm_landmarks) > 0:
            self.draw_lines_between_models(comparison_frame, mp_landmarks, torchlm_landmarks)

        # Stack frames
        combined_frame = np.vstack((np.hstack((original_frame, mp_frame)), np.hstack((torchlm_frame, comparison_frame))))

        return combined_frame

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=640)
            processed_frame = self.create_orientation(frame)
            out.write(processed_frame)

        cap.release()
        out.release()
        return 'output.mp4'

face_models_comparison = FaceModelsComparison()

def process_image(image):
    result = face_models_comparison.create_orientation(image)
    return result

def process_video(video):
    output_path = face_models_comparison.process_video(video)
    return output_path

# Define the Gradio interface
models_comparison_interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Image for Models Comparison"),
    outputs=gr.Image(type="numpy", label="Models Comparison Result"),
    live=True,
    title="Models Comparison",
    description="Compare face landmark detection results from MediaPipe and Torchlm models."
)

video_comparison_interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video for Models Comparison"),
    outputs=gr.Video(label="Processed Video (需要下载查看)"),
    title="Models Comparison - Video",
    description="Compare face landmark detection results from MediaPipe and Torchlm models on video."
)

# Combine interfaces into a single tabbed interface
iface = gr.TabbedInterface(
    [models_comparison_interface, video_comparison_interface],
    ["Image Processing", "Video Processing"]
)

iface.launch(share=True)
