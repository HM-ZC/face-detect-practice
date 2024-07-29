import gradio as gr
import cv2
import dlib
import numpy as np
from PIL import Image
import os


class FaceOrientation(object):
    def __init__(self):
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

    def overlay_logo(self, frame):
        logo_image = cv2.imread('../66b29fbce5d85d93186c2d5aacfa70f.jpg')
        if logo_image is None:
            raise ValueError("无法加载校徽图像")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)

        for subject in subjects:
            landmarks = self.predict(gray, subject)
            size = frame.shape

            # 获取脸部特征点
            points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 68)], dtype=np.int32)

            # 选择四个点进行透视变换
            src_points = np.float32([
                [0, 0],
                [logo_image.shape[1] - 1, 0],
                [logo_image.shape[1] - 1, logo_image.shape[0] - 1],
                [0, logo_image.shape[0] - 1]
            ])

            # 获取脸部的四个点，近似为矩形
            dst_points = np.float32([
                [landmarks.part(36).x, landmarks.part(36).y],  # 左眼左角
                [landmarks.part(45).x, landmarks.part(45).y],  # 右眼右角
                [landmarks.part(54).x, landmarks.part(54).y],  # 右嘴角
                [landmarks.part(48).x, landmarks.part(48).y]  # 左嘴角
            ])

            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_logo = cv2.warpPerspective(logo_image, M, (frame.shape[1], frame.shape[0]))

            # 叠加校徽图像到原始帧
            frame[warped_logo > 0] = warped_logo[warped_logo > 0]

        return frame


face_orientation = FaceOrientation()


def process_image(image):
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = face_orientation.overlay_logo(frame)
    return cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = face_orientation.overlay_logo(frame)
        frames.append(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
    cap.release()

    # 保存处理后的视频
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    return output_path  # 只返回视频文件路径


image_interface = gr.Interface(fn=process_image, inputs=gr.Image(), outputs=gr.Image(), live=True)
video_interface = gr.Interface(fn=process_video, inputs=gr.Video(), outputs=gr.Video(label="需要下载查看结果"))

iface = gr.TabbedInterface([image_interface, video_interface], ["处理单个图像", "处理视频"])

# 启动 Gradio 应用并获取公网 URL
iface.launch(share=True)