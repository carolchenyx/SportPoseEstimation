from .opt import opt
from .pPose_nms import pose_nms
from SPPE.src.utils.eval import getPrediction
import numpy as np
from app_class.exercise_src.function.locator import Locator as PULocator

if opt.vis_fast:
    from .fn import vis_frame_fast as vis_frame, vis_frame_black
else:
    from .fn import vis_frame, vis_frame_black


class PoseEstimator(object):
    def __init__(self, frameSize=(640,480)):
        self.final_result = []
        self.img = []
        self.skeleton = []
        self.Locator = ''
        # initialize the queue used to store frames read from
        # the video file
        self.cnt = 0
        # self.log = open("log/DataWriter.txt", "w")
        self.result = []

    def process(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
        orig_img = np.array(orig_img, dtype=np.uint8)
        if boxes is None:
            return orig_img, []
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            preds_hm, preds_img, preds_scores = getPrediction(
                hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            self.result = pose_nms(boxes, scores, preds_img, preds_scores)
            if self.result:
                result = self.locate()
                # result = self.result
                result = {
                    'imgname': im_name,
                    'result': result
                }
                self.final_result.append(result)
                img_black, pred_black = vis_frame_black(orig_img, result)
                self.img = img_black
                self.skeleton = pred_black
                img, pred = vis_frame(orig_img, result)
                return img, self.skeleton, self.img
            else:
                return orig_img, [], orig_img

    def locate(self):
        return self.result

#
# class PoseEstimatorGolf(PoseEstimator):
#     def __init__(self):
#         super().__init__()
#         self.Locator = GLocator()
#
#     def locate(self):
#         return [self.Locator.locate_user(self.result)]


class PoseEstimatorYoga(PoseEstimator):
    def __init__(self):
        super().__init__()


class PoseEstimatorPushUp(PoseEstimator):
    def __init__(self):
        super().__init__()
        self.Locator = PULocator()

    def locate(self):
        return [self.Locator.locate_user(self.result)]
