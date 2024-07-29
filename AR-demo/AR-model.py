import cv2
import dlib
import numpy as np

class FaceOrientation(object):
    def __init__(self):
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

    def create_orientation(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        logo_image = cv2.imread('../66b29fbce5d85d93186c2d5aacfa70f.jpg', cv2.IMREAD_UNCHANGED)
        if logo_image is None:
            print("无法加载校徽图像")
            return

        # 检查校徽图像是否具有4个通道（RGBA），如果没有，添加一个Alpha通道
        if logo_image.shape[2] == 3:
            logo_image = cv2.cvtColor(logo_image, cv2.COLOR_BGR2BGRA)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = self.detect(gray, 0)

            for subject in subjects:
                landmarks = self.predict(gray, subject)
                size = frame.shape

                # 获取脸部特征点
                points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 68)], dtype=np.int32)

                # 计算包含脸部特征点的矩形区域
                (x, y, w, h) = cv2.boundingRect(points)

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
                    [landmarks.part(48).x, landmarks.part(48).y]   # 左嘴角
                ])

                # 计算透视变换矩阵
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                warped_logo = cv2.warpPerspective(logo_image, M, (frame.shape[1], frame.shape[0]))

                # 叠加校徽图像到原始帧
                mask = warped_logo[..., 3:] / 255.0  # 获取Alpha通道
                for c in range(3):
                    frame[..., c] = frame[..., c] * (1 - mask[..., 0]) + warped_logo[..., c] * mask[..., 0]

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    face_obj = FaceOrientation()
    face_obj.create_orientation()