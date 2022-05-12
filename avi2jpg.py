from PIL import Image
import cv2


def spilt_frame(vedio_data, save_path):
    cap = cv2.VideoCapture(vedio_data)
    num = 1

    while True:
        success, data = cap.read()
        if not success:
            break
        img_bgr = Image.fromarray(data)
        # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if num % 500 == 0:
            name = str(num)
            img_bgr.save(save_path + name + ".jpg")

        num = num + 1


if __name__ == '__main__':
    # distance = None
    avi = "your_avi.mp4"
    root = "path/to/your/avi"
    save_path = root
    vedio = root + avi
    spilt_frame(vedio, save_path)
