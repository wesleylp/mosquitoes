import sys

from video_utils import videoObj

sys.path.append('../codes')

video_path = '../data/VideoDataSet/5m/DJI00804_05m_04aTomada.mp4'
annot_path = '../data/zframer-marcacoes/Teste/5m/DJI00804_05m_04aTomada.txt'

video = videoObj(videopath=video_path, annotation_path=annot_path)
video.videoInfo.printAllInformation()
# video._annotation.is_valid()

video.play_video(show_bb=True)
