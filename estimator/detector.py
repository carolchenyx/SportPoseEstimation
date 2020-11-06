import torch
from SPPE.src.utils.img import cropBox, im_to_torch
from .opt import opt
from yolo.preprocess import prep_frame
from yolo.util import dynamic_write_results
from yolo.darknet import Darknet
import cv2


class VideoProcessor(object):
    def __init__(self):
        self.in_dim = int(opt.inp_dim)

    def process(self, frame):
        img = []
        orig_img = []
        im_name = []
        im_dim_list = []
        img_k, orig_img_k, im_dim_list_k = prep_frame(frame, self.in_dim)

        img.append(img_k)
        orig_img.append(orig_img_k)
        im_name.append('0.jpg')
        im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
            im_dim_list_ = im_dim_list

        return img, orig_img, im_name, im_dim_list

class ObjectDetection(object):
    def __init__(self, batchSize=1):
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.eval()

        self.stopped = False
        self.batchSize = batchSize

    def process(self, img, orig_img, im_name, im_dim_list):
        with torch.no_grad():
            # Human Detection
            img = img
            prediction = self.det_model(img, CUDA=False)
            # NMS process
            dets = dynamic_write_results(prediction, opt.confidence,  opt.num_classes, nms=True, nms_conf=opt.nms_thesh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                return orig_img[0], im_name[0], None, None, None, None, None

            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return orig_img[0], im_name[0], None, None, None, None, None
        inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
        pt1 = torch.zeros(boxes_k.size(0), 2)
        pt2 = torch.zeros(boxes_k.size(0), 2)
        return orig_img[0], im_name[0], boxes_k, scores[dets[:, 0] == 0], inps, pt1, pt2


class DetectionProcessor(object):
    def __init__(self):
        self.cnt = 0

    def process(self, orig_img, im_name, boxes, scores, inps, pt1, pt2):
        with torch.no_grad():
            if orig_img is None:
                return None, None, None, None, None, None, None

            if boxes is None or boxes.nelement() == 0:
                return None, orig_img, im_name, boxes, scores, None, None

            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
            return inps, orig_img, im_name, boxes, scores, pt1, pt2


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor((float(box[0]), float(box[1])))
        bottomRight = torch.Tensor((float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2