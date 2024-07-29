import cv2
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from torchlm.runtime import faceboxesv2_ort, pipnet_ort

def real_time_face_detection():
    device = "cpu"
    checkpoint = "../model/pipnet_resnet101_10x68x32x256_300w.pth"

    # 加载模型
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
            map_location=device,
            checkpoint=checkpoint
        )
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 进行人脸检测和关键点识别
        landmarks, bboxes = torchlm.runtime.forward(frame)
        frame = torchlm.utils.draw_bboxes(frame, bboxes=bboxes)
        frame = torchlm.utils.draw_landmarks(frame, landmarks=landmarks)

        # 显示结果
        cv2.imshow('Real-Time Face Detection', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_detection()