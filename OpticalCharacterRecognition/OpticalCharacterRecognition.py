from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
from PIL import Image

# 初始化PaddleOCR，设置需要的语言
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 支持的语言可以是 'ch', 'en', 'japan', 'korean' 等

# 指定图片路径
image_path = r"D:\Photo\self_photo\Screenshot_20210914_222231_com.huawei.android.launcher.jpg"

# 进行OCR识别
result = ocr.ocr(image_path, cls=True)

# 打印识别结果
for line in result[0]:
    print(f"Text: {line[1][0]} Confidence: {line[1][1]}")

# 画出识别结果
image = Image.open(image_path).convert('RGB')
boxes = [elements[0] for elements in result[0]]
pairs = [elements[1] for elements in result[0]]
txts = [pair[0] for pair in pairs]
scores = [pair[1] for pair in pairs]
im_show = draw_ocr(image, boxes, txts, scores, font_path='C:\Windows\Fonts\simsun.ttc')  # 设置合适的字体路径

# 显示结果
plt.imshow(im_show)
plt.axis('off')
plt.show()



import easyocr
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import rcParams

# 配置 Matplotlib 使用支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 适用于中文
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示问题

# 初始化 EasyOCR 读者
reader = easyocr.Reader(['en', 'ch_sim'])

# 指定图像路径
image_path = r"D:\Photo\self_photo\Screenshot_20210914_222231_com.huawei.android.launcher.jpg"

# 使用 EasyOCR 进行文本识别
results = reader.readtext(image_path)

# 打印识别结果及置信度
print("Recognized Text:")
for bbox, text, score in results:
    print(f"Text: {text}; Confidence: {score:.2f}")

# 加载图像
image = Image.open(image_path)

# 创建图形和轴
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# 显示原始图像
ax[0].imshow(image)
ax[0].axis('off')  # 不显示坐标轴

# 绘制识别框
for bbox, text, score in results:
    bbox = [tuple(point) for point in bbox]
    rect = Polygon(bbox, edgecolor='red', facecolor='none', linewidth=2)
    ax[0].add_patch(rect)

# 显示识别结果
text_str = '\n'.join([f'{text} (Confidence: {score:.2f})' for _, text, score in results])
ax[1].text(0.3, 1.0, text_str, fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))

# 设置文本图的属性
ax[1].axis('off')  # 不显示坐标轴
ax[1].set_title('Detected Text', fontsize=16)

# 设置图像标题
ax[0].set_title('Detected Text Boxes', fontsize=16)

# 显示图像
plt.tight_layout()
plt.show()
