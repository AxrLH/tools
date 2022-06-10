import cv2


def video_capture():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    # 获取视频帧数
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # 获取视频编码
    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc2 = int(cap2.get(cv2.CAP_PROP_FOURCC))

    # cv2.namedWindow("cap1", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("cap2", cv2.WINDOW_AUTOSIZE)

    path1 = "F:/自动驾驶数据库/红外自测数据/艾睿光电/艾睿数据采集/视频录制/20220518/USB接口13mm/" + "cam1_13mm" + ".avi"
    path2 = "F:/自动驾驶数据库/红外自测数据/艾睿光电/艾睿数据采集/视频录制/20220518/USB接口9.1mm/" + "cam2_9mm" + ".avi"

    #写入视频
    out1 = cv2.VideoWriter(path1, fourcc1, fps1, (640, 512))
    out2 = cv2.VideoWriter(path2, fourcc2, fps2, (640, 512))

    while cap1.isOpened() and cap2.isOpened():
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()

        cv2.imshow("cam1_13mm", frame1)
        cv2.imshow("cam2_9mm", frame2)

        key = cv2.waitKey(1)

        out1.write(frame1)
        out2.write(frame2)

        if key == ord('q'):
            break
    cap1.release()
    cap2.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_capture()
