import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from stgcn.stgcn import STGCN
from PIL import Image, ImageDraw, ImageFont

mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

KEY_JOINTS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

POSE_CONNECTIONS = [(6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1), (12, 10),
                    (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0)]

POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
               (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
               (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

ACTION_MODEL_MAX_FRAMES = 30

class FallDetection:
    def __init__(self):
        self.action_model = STGCN(weight_file='./weights/tsstg-model.pth', device='cpu')
        self.joints_list = deque(maxlen=ACTION_MODEL_MAX_FRAMES)

    def draw_skeleton(self, frame, pts):
        l_pair = POSE_CONNECTIONS
        p_color = POINT_COLORS
        line_color = LINE_COLORS

        part_line = {}
        pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
        for n in range(pts.shape[0]):
            if pts[n, 2] <= 0.05:
                continue
            cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)
            # cv2.putText(frame, str(n), (cor_x+10, cor_y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(frame, start_xy, end_xy, line_color[i], int(1*(pts[start_p, 2] + pts[end_p, 2]) + 3))
        return frame

    def cv2_add_chinese_text(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/MSYH.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def detect(self):
        # Initialize the webcam capture.
        cap = cv2.VideoCapture("./video2.mp4")
        image_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        image_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_num = 0
        print(image_h, image_w)

        with mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                fps_time = time.time()
                frame_num += 1
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # 提高性能
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue

                # 识别骨骼点
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # mp_drawing.draw_landmarks(
                #     image,
                #     results.pose_landmarks,
                #     mp_pose.POSE_CONNECTIONS,
                #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                landmarks = results.pose_landmarks.landmark
                joints = np.array([[landmarks[joint].x * image_w,
                                    landmarks[joint].y * image_h,
                                    landmarks[joint].visibility]
                                   for joint in KEY_JOINTS])
                # 人体框
                box_l, box_r = int(joints[:, 0].min())-50, int(joints[:, 0].max())+50
                box_t, box_b = int(joints[:, 1].min())-100, int(joints[:, 1].max())+100

                self.joints_list.append(joints)

                # 识别动作
                action = ''
                clr = (0, 255, 0)
                # 30帧数据预测动作类型
                if len(self.joints_list) == ACTION_MODEL_MAX_FRAMES:
                    pts = np.array(self.joints_list, dtype=np.float32)
                    out = self.action_model.predict(pts, (image_w, image_h))
                    action_name = self.action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    print(action)

                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)
                        action = '摔倒'
                    elif action_name == 'Walking':
                        clr = (255, 128, 0)
                        action = '行走'
                    else:
                        action = ''

                # 绘制骨骼点和动作类别
                image = self.draw_skeleton(image, self.joints_list[-1])
                image = cv2.rectangle(image, (box_l, box_t), (box_r, box_b), (255, 255, 0), 2)
                image = self.cv2_add_chinese_text(image, f'状态：{action}', (box_l + 10, box_t + 10), clr, 80)
                image = cv2.putText(image, f'FPS: {int(1.0 / (time.time() - fps_time))}',
                                    (50, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
                # Flip the image horizontally for a selfie-view display.

                down_width = 1400
                down_height = 800
                down_points = (down_width, down_height)
                image = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)

                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release the resources.
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    FallDetection().detect()



