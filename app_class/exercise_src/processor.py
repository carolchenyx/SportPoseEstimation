from estimator.detector import ObjectDetection, DetectionProcessor, VideoProcessor
from estimator.pose_estimator import PoseEstimatorPushUp as PoseEstimator
from estimator.datatset import Mscoco
from estimator.opt import opt
from SPPE.src.main_fast_inference import *
from config import config

args = opt
args.dataset = 'coco'

step = config.golf_static_step
batchSize = args.posebatch


class ImageProcessor(object):
    def __init__(self):
        self.object_detector = ObjectDetection()
        self.detection_processor = DetectionProcessor()
        self.data_recorder = PoseEstimator()
        self.video_processor = VideoProcessor()

        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)

        self.pose_model.eval()
        self.pose_model.cuda()

    def process_img(self, frame):
        img, orig_img, im_name, im_dim_list = self.video_processor.process(frame)

        with torch.no_grad():
            inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.object_detector.process(img, orig_img, im_name,
                                                                                            im_dim_list)
            inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.detection_processor.process(inps, orig_img, im_name,
                                                                                                boxes, scores, pt1, pt2)
            # try:
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []

            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm).cpu().data
            ske_img, skeleton, ske_black_img = self.data_recorder.process(boxes, scores, hm, pt1, pt2, orig_img,
                                                                          im_name.split('/')[-1])
            return skeleton, ske_img, ske_black_img

            # except:
            #     return [], frame, frame
#

# from src.detector.visualize import BBoxVisualizer
# from src.tracker.visualize import IDVisualizer
# from src.estimator.visualize import KeyPointVisualizer
# import cv2
# # from config.config import video_path, frame_size
# from utils.utils import str2boxdict, str2kpsdict, str2kpsScoredict
#
# frame_size = (720,540)
# body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
#               "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
#               "Left ankle", "Right ankle"]
# body_dict = {name: idx for idx, name in enumerate(body_parts)}
#
#
# class ImageProcessor(object):
#     def __init__(self, video_path):
#         # self.BBV = BBoxVisualizer()
#         self.cap = cv2.VideoCapture(video_path)
#         self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         with open("/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_box.txt", "r") as bf:
#             self.box_txt = [line[:-1] for line in bf.readlines()]
#         with open("/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps.txt", "r") as kf:
#             self.kps_txt = [line[:-1] for line in kf.readlines()]
#         with open("/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] +
#                   "_kps_score.txt", "r") as ksf:
#             self.kps_score_txt = [line[:-1] for line in ksf.readlines()]
#         self.IDV = IDVisualizer(with_bbox=True)
#         self.KPV = KeyPointVisualizer()
#
#     def process_img(self):
#         cnt = 0
#         while True:
#             ret, frame = self.cap.read()
#             frame = cv2.resize(frame, frame_size)
#             cnt += 1
#             if ret:
#                 id2bbox = str2boxdict(self.box_txt.pop(0))
#                 id2kps = str2kpsdict(self.kps_txt.pop(0))
#                 id2kpsScore = str2kpsScoredict(self.kps_score_txt.pop(0))
#                 if id2bbox is not None:
#                     frame = self.IDV.plot_bbox_id(id2bbox, frame)
#                 if id2kps is not None:
#                     kps_tensor, score_tensor = self.KPV.kpsdic2tensor(id2kps, id2kpsScore)
#                     frame = self.KPV.vis_ske(frame, kps_tensor, score_tensor)
#                 cv2.imshow("res", frame)
#                 cv2.waitKey(500)
#             else:
#                 self.cap.release()
#                 cv2.destroyAllWindows()
#                 break
#
#
# if __name__ == '__main__':
#     # src_folder = "video/push_up"
#     # dest_fodler = src_folder + "kps"
#     # sub_folder = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder)]
#     # sub_dest_folder = [os.path.join(dest_fodler, folder) for folder in os.listdir(src_folder)]
#
#     VideoProcessor(video_path).process_video()
