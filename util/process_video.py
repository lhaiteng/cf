# -*- coding: utf-8 -*-
import cv2, os


class ProcessVideo:
    def grap_pic(self, save_dir, src_path, num_pics=0, reshape=0, pre=''):
        """
        :param save_dir: 图片保存文件夹
        :param src_path: 来源视频路径
        :param num_pics: 抓取的图片数
        :param reshape: 抓取到的图片放缩到的形状。默认0是不放缩。
            若类型是int或float是同比放缩，若类型是list或tuple是放缩到指定长宽。
        :param pre:
        :return:
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        gp_iter = self.grab_pic_iter(src_path, num_pics=num_pics, reshape=reshape)

        print(f'START grabbing {num_pics} pic from {src_path}')
        save_path = save_dir + pre + '{ind_pic}.png'
        ind = -1
        while 1:
            ind += 1
            try:
                pic = next(gp_iter)
                cv2.imwrite(save_path.format(ind_pic=ind), pic)
            except:
                print(f'FINISH grabbing {ind + 1} pics from {src_path}')
                break

    def grab_pic_iter(self, src_path, num_pics=0, reshape=0):
        # 提取视频的fps、宽、高
        cap = cv2.VideoCapture(src_path)
        success = cap.isOpened()
        if success:
            # video_width = cap.get()
            # video_height = cap.get()
            rate = cap.get(5)  # 获取帧率
            fraNum = cap.get(7)  # 获取帧数
        else:
            raise ValueError(f'ERROR cannot open {src_path}')
        if num_pics:
            per_frame = fraNum // num_pics
        else:
            per_frame = 1
        num_frame = -1
        while success:
            num_frame += 1
            success, frame = cap.read()  # 得到每一帧图片frame
            if reshape:
                if type(reshape) in (int, float):
                    frame = cv2.resize(frame, (0, 0), fx=reshape, fy=reshape)
                elif type(reshape) in (tuple, list):
                    frame = cv2.resize(frame, reshape)
            if not success:
                print(f"FAILED  read x new frame from {src_path}.")
                break
            if num_frame % per_frame == 0:
                yield frame
            else:
                continue

    def write_video_from_video(self, out_path, src_video, fps=24):
        """
        四个字符用来表示压缩帧的codec 例如：
        CV_FOURCC('P','I','M','1') = MPEG-1 codec
        CV_FOURCC('M','J','P','G') = motion-jpeg codec  --> mp4v
        CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
        CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
        CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec  --> avi
        CV_FOURCC('U', '2', '6', '3') = H263 codec
        CV_FOURCC('I', '2', '6', '3') = H263I codec
        CV_FOURCC('F', 'l', 'V', '1') = FLV1 codec
        NOTE：生成文件占用空间最小的编码方式是MPEG-4.2 codec。
        在VideoWriter类的构造函数参数为CV_FOURCC('M', 'P', '4', '2') 。
        最大的是MPEG-1 codec，
        对应在VideoWriter类的构造函数参数为CV_FOURCC('P','I','M','1') ，
        所占磁盘空间是前者的5.7倍。
        所以如果需要24小时全天候录制监控，可以优先使用MPEG-4.2的编解码方式。
        """
        # 限定输出视频的编码格式，不同的编码格式所对应的存储空间差别较大，可以根据自己的需求改变对应的编码格式
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', '2')
        # 将提前设置好的需要对视频做更改的参数输入，构建一个视频输出流接口，用于对视频帧进行改变并重新写成视频。
        out_shape = (128, 128)
        videoWriter = cv2.VideoWriter(out_path, fourcc, fps, out_shape)

        # 对视频的每一帧进行读取和相应的操作
        # 若是改变视频速度的话，已经在构建videoWriter的时候通过fps限定了，那么在该过程中只需要直接写入视频帧即可。
        pic_iter = self.grab_pic_iter(src_video)
        print(f'START writing video from video: {src_video} into video: {out_path}...')
        while 1:
            try:
                frame = next(pic_iter)
                frame = cv2.resize(frame, out_shape, cv2.INTER_AREA)
                videoWriter.write(frame)
            except:
                print(f'FINISH writing video from video: {src_video} into video: {out_path}.')
                break

        videoWriter.release()

    def write_video_from_pics(self, out_path, pic_dir, fps=24, video_time=0):
        if os.path.isdir(pic_dir):
            pic_names = os.listdir(pic_dir)
            pic_paths = [pic_dir + n for n in pic_names if os.path.isfile(pic_dir + n)]
        else:
            raise FileExistsError(f'ERROR {pic_dir} not exists.')

        num_pics = len(pic_paths)

        if video_time:
            fps = video_time / num_pics

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', '2')
        out_shape = (128, 128)
        videoWriter = cv2.VideoWriter(out_path, fourcc, fps, out_shape)

        print(f'START writing video from pic_dir:{pic_dir} into video:{out_path}...')
        for _path in pic_paths:
            pic = cv2.imread(_path)
            pic = cv2.resize(pic, out_shape, cv2.INTER_AREA)  # 缩小
            videoWriter.write(pic)
        videoWriter.release()
        print(f'FINISH writing video from pic_dir:{pic_dir} into video:{out_path}.')


if __name__ == '__main__':
    pv = ProcessVideo()

    src_path = "E:/TEST/AI/datasets/changeface_video/a1.mp4"
    out_path = "E:/TEST/AI/datasets/changeface_video/a_out.mp4"
    # pv.write_video_from_video(out_path, src_video=src_path)

    """从视频中提取指定数目的图片"""
    # num_pics = 500
    # save_dir = 'E:/TEST/AI/datasets/changeface_video/a1/'
    # pv.grap_pic(save_dir, src_path, num_pics=num_pics, reshape=(533, 300), pre='x')
    # num_pics = 1000
    # save_dir = 'E:/TEST/AI/datasets/changeface_video/a2/'
    # pv.grap_pic(save_dir, src_path, num_pics=num_pics, reshape=(533, 300), pre='y')
    num_pics = 150
    save_dir = 'E:/TEST/AI/datasets/changeface_video/shuilaili/'
    src_path = "E:/TEST/AI/datasets/changeface_video/shuilaili.mp4"
    pv.grap_pic(save_dir, src_path, num_pics=num_pics, reshape=0, pre='x')

    # pic_dir = 'E:/TEST/AI/datasets/changeface_video/x/'
    # pv.write_video_from_pics(out_path, pic_dir)
