import cv2
import dlib
import imutils
import numpy as np

class FaceOrientation:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

    def create_orientation(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = self.detector(gray, 0)

            for subject in subjects:
                landmarks = self.predictor(gray, subject)
                size = frame.shape

                # 2D image points
                image_points = np.array([
                    (landmarks.part(33).x, landmarks.part(33).y),  # Nose tip
                    (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                    (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                    (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
                    (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                    (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
                ], dtype="double")

                # 3D model points
                model_points = np.array([
                    (0.0, 0.0, 0.0),  # Nose tip
                    (0.0, -330.0, -65.0),  # Chin
                    (-225.0, 170.0, -135.0),  # Left eye left corner
                    (225.0, 170.0, -135.0),  # Right eye right corner
                    (-150.0, -150.0, -125.0),  # Left Mouth corner
                    (150.0, -150.0, -125.0)  # Right mouth corner
                ])

                # Camera internals
                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double"
                )

                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs)

                (b1, _) = cv2.projectPoints(
                    np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                (b2, _) = cv2.projectPoints(
                    np.array([(-350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                (b3, _) = cv2.projectPoints(
                    np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                (b4, _) = cv2.projectPoints(
                    np.array([(350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                (b11, _) = cv2.projectPoints(
                    np.array([(450.0, 350.0, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                (b12, _) = cv2.projectPoints(
                    np.array([(-450.0, -350.0, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                (b13, _) = cv2.projectPoints(
                    np.array([(-450.0, 350, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                (b14, _) = cv2.projectPoints(
                    np.array([(450.0, -350.0, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                b1 = (int(b1[0][0][0]), int(b1[0][0][1]))
                b2 = (int(b2[0][0][0]), int(b2[0][0][1]))
                b3 = (int(b3[0][0][0]), int(b3[0][0][1]))
                b4 = (int(b4[0][0][0]), int(b4[0][0][1]))

                b11 = (int(b11[0][0][0]), int(b11[0][0][1]))
                b12 = (int(b12[0][0][0]), int(b12[0][0][1]))
                b13 = (int(b13[0][0][0]), int(b13[0][0][1]))
                b14 = (int(b14[0][0][0]), int(b14[0][0][1]))

                # Draw rectangles and lines
                cv2.line(frame, b1, b3, (255, 255, 0), 2)
                cv2.line(frame, b3, b2, (255, 255, 0), 2)
                cv2.line(frame, b2, b4, (255, 255, 0), 2)
                cv2.line(frame, b4, b1, (255, 255, 0), 2)

                cv2.line(frame, b11, b13, (255, 255, 0), 2)
                cv2.line(frame, b13, b12, (255, 255, 0), 2)
                cv2.line(frame, b12, b14, (255, 255, 0), 2)
                cv2.line(frame, b14, b11, (255, 255, 0), 2)

                cv2.line(frame, b11, b1, (0, 255, 0), 2)
                cv2.line(frame, b13, b3, (0, 255, 0), 2)
                cv2.line(frame, b12, b2, (0, 255, 0), 2)
                cv2.line(frame, b14, b4, (0, 255, 0), 2)

            # Create additional frames
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges_frame = cv2.Canny(frame, 100, 200)

            # Resize frames to fit the screen width
            frame = cv2.resize(frame, (640, 480))
            gray_frame = cv2.resize(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR), (640, 480))
            edges_frame = cv2.resize(cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2BGR), (640, 480))

            # Stack frames vertically
            combined_frame = np.hstack((frame, gray_frame, edges_frame))

            cv2.imshow("Combined Frame", combined_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':
    face_orientation = FaceOrientation()
    face_orientation.create_orientation()