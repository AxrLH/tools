import cv2
import os


# TODO:无法在win10下安装video decoder，没找到合适的包，在Ubuntu18.04下可转可用
def img2vedio(img_size, img_dir, video_path, fps):
    format = cv2.VideoWriter_fourcc(*'XVID')
    video_write = cv2.VideoWriter(video_path, format, fps, img_size)

    for idx in (os.listdir(img_dir)):
        # img = os.path.join(img_dir, idx)
        frame = cv2.imread(img_dir + idx)
        # print(frame.shape)

        video_write.write(frame)

    video_write.release()
    print(f'finish changing, saving in {video_path}')


if __name__ == '__main__':
    img_dir = 'path/to/your/img'
    par_path = os.path.dirname(img_dir)
    file = os.path.split(img_dir)[-1]
    filename = 'out.avi'
    video_path = 'path/to/your/save/video.avi'

    fps = 3.0
    img_size = (640, 512)
    img2vedio(img_size, img_dir, video_path, fps)
