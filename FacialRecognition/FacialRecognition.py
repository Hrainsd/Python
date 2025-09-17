# 人脸验证和人脸属性分析
from deepface import DeepFace
import cv2
import time

# 定义两个图片的路径
img1_path = "D:\Photo\self_photo\IMG_20230818_121818.jpg"
img2_path = "D:\Photo\self_photo\IMG_20240115_132327.jpg"

# 打印输入的图像路径
print("验证图像 1 路径:", img1_path)
print("验证图像 2 路径:", img2_path)

# 开始计时
start_time = time.time()

# 选择模型
# Google FaceNet ('Facenet')
# DeepFace ('DeepFace')
# VGG-Face ('VGG-Face')
# OpenFace ('OpenFace')
model_name = 'Facenet'

# 进行人脸验证
result = DeepFace.verify(img1_path, img2_path, model_name=model_name, enforce_detection=False)

# 打印验证结果
print("是否是同一人:", result["verified"])
print("相似度 (距离):", result["distance"])
print("使用的模型:", result["model"])

# 计算并打印处理时间
end_time = time.time()
print("处理时间: {:.2f} 秒".format(end_time - start_time))


# 定义图片的路径列表
img_paths = [
    "D:\\Photo\\self_photo\\mmexport1645876781739.jpg",
    "D:\Photo\self_photo\IMG_20210918_194019.jpg",
    "D:\Photo\self_photo\IMG_20210918_193948.jpg",
    "D:\Photo\self_photo\IMG_20210613_133556.jpg"
]

# 遍历每张图片进行分析
for img_path in img_paths:
    try:
        # 进行人脸属性分析
        analysis = DeepFace.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        # 打印分析结果
        print(f"分析结果 - {img_path}:")
        print("年龄:", analysis[0]["age"])
        print("性别:", analysis[0]["gender"])
        print("种族:", analysis[0]["dominant_race"])
        print("情绪:", analysis[0]["dominant_emotion"])
        print("\n")  # 打印空行以便分隔不同图片的结果

    except Exception as e:
        print(f"分析 {img_path} 时出错: {str(e)}")



# 实时人脸识别
from deepface import DeepFace
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 已知的脸部图像路径和对应的名字
known_images = {
    "xxx": "D:\\Photo\\self_photo\\IMG_20230818_121818.jpg",
    "yyy": "D:\\Photo\\self_photo\\IMG_20220629_182747.jpg"
}

# 选择模型
model_name = 'Facenet'

# 开始捕获视频流
cap = cv2.VideoCapture(0)  # 使用摄像头，参数0表示默认摄像头

# 初始化计时器
start_time = time.time()

# 使用的字体路径，需确保系统中有该字体
font_path = "C:\\Windows\\Fonts\\simsun.ttc"  # SimSun 字体路径


def draw_text(image, text, position, font, color):
    """在图像上绘制多行文本"""
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=color)
    return image


while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()

    if not ret:
        print("未能读取到视频流")
        break

    # 将 OpenCV 图像转换为 Pillow 图像
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    try:
        analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        recognized = False

        for person_name, image_path in known_images.items():
            result = DeepFace.verify(frame, image_path, model_name=model_name, enforce_detection=False)
            if result["verified"]:
                # 提取性别判断结果
                gender_scores = analysis[0]['gender']
                if gender_scores['Man'] > gender_scores['Woman']:
                    gender = '男'
                else:
                    gender = '女'

                # 创建要显示的文本
                lines = [
                    f"姓名: {person_name}",
                    f"性别: {gender}",
                    f"情绪: {analysis[0]['dominant_emotion']}"
                ]

                # 使用 PIL 绘制文本
                font = ImageFont.truetype(font_path, 24)  # 使用 SimSun 字体
                y0, dy = 50, 30
                for i, line in enumerate(lines):
                    y = y0 + i * dy
                    frame_pil = draw_text(frame_pil, line, (50, y), font, (0, 0, 0))

                recognized = True
                break

        if not recognized:
            frame_pil = draw_text(frame_pil, "Unknown Person", (50, 50), font, (0, 0, 255))

    except Exception as e:
        print(f"识别时出错: {str(e)}")

    # 将 Pillow 图像转换回 OpenCV 图像
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 显示结果
    cv2.imshow('Real-time Face Recognition', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()

# 计算并打印处理时间
end_time = time.time()
print("处理时间: {:.2f} 秒".format(end_time - start_time))



# 视频人脸识别
from deepface import DeepFace
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 已知的脸部图像路径和对应的名字
known_images = {
    "xxx": "D:\\Photo\\self_photo\\IMG_20230818_121818.jpg",
    "yyy": "D:\\Photo\\self_photo\\IMG_20220629_182747.jpg",
    "zzz": "D:\\Photo\\self_photo\\IMG_20210829_161036.jpg"
}

# 选择模型
model_name = 'Facenet'

# 视频文件路径
input_video_path = "C:\\Users\\23991\\OneDrive\\视频\\video\\VID_20240118_215912.mp4"
output_video_path = "C:\\Users\\23991\\OneDrive\\视频\\video\\1_cv_output.mp4"

# 打开视频文件
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# 视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义输出视频文件的编解码器和视频流
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 使用的字体路径，需确保系统中有该字体
font_path = "C:\\Windows\\Fonts\\simsun.ttc"

def draw_text_on_frame(frame, text, position, font, color):
    """在 OpenCV 图像上绘制文本"""
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 存储最新的文本信息
current_text = ""
frame_counter = 0
frame_interval = int(fps)  # 每秒处理一次
display_duration = int(fps * 0.5)  # 0.5秒显示文本

# 用于控制文本显示的计数器
display_counter = 0

while True:
    # 从视频文件读取一帧
    ret, frame = cap.read()

    if not ret:
        break  # 如果没有更多帧，则退出循环

    if frame_counter % frame_interval == 0:
        try:
            print(f"Processing frame {frame_counter}")

            # 进行人脸属性分析
            analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            recognized = False

            for person_name, image_path in known_images.items():
                result = DeepFace.verify(frame, image_path, model_name=model_name, enforce_detection=False)
                if result["verified"]:
                    # 提取性别判断结果
                    gender_scores = analysis[0]['gender']
                    gender = '男' if gender_scores['Man'] > gender_scores['Woman'] else '女'

                    print(f"Recognized {person_name} 性别: {gender} 情绪: {analysis[0]['dominant_emotion']}")

                    # 创建要显示的文本
                    lines = [
                        f"姓名: {person_name}",
                        f"性别: {gender}",
                        f"情绪: {analysis[0]['dominant_emotion']}"
                    ]

                    # 更新当前文本
                    current_text = "\n".join(lines)
                    display_counter = 0  # 重置显示计数器
                    recognized = True
                    break

            if not recognized:
                current_text = "Unknown Person"
                display_counter = 0  # 重置显示计数器

        except Exception as e:
            print(f"识别时出错: {str(e)}")
            current_text = "Error"
            display_counter = 0  # 重置显示计数器

    # 如果文本显示计数器小于显示持续时间，则绘制文本
    if display_counter < display_duration:
        font = ImageFont.truetype(font_path, 50)
        y0, dy = 0, 50
        for i, line in enumerate(current_text.split('\n')):
            y = y0 + i * dy
            frame = draw_text_on_frame(frame, line, (50, y), font, (0, 0, 0))
        display_counter += 1  # 增加显示计数器

    # 写入输出视频文件
    out.write(frame)
    frame_counter += 1

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
