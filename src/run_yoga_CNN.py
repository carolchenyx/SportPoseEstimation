from config import config
from estimator.opt import opt
from estimator.dataloader import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from SPPE.src.main_fast_inference import *
from app_class.yoga_src.models.CNN_model import CNN_model
import sys
import cv2
from app_class.yoga_src.analyzer import PostureAnalyser

args = opt
args.dataset = 'coco'
mode = args.mode
batchSize = args.posebatch

video_path = config.yoga_video_path
CNN_path = config.CNN_yoga_model
CNN_pre_train = config.CNN_yoga_pre_train_model
CNN_class = config.CNN_yoga_class
CNN_pre_train_path = "models/pre_train_model/{}.pth".format(CNN_pre_train)
class_num = len(CNN_class)
nece_point = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

args.sp = True
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def run(video_path):
    # Load input video
    data_loader = VideoLoader(video_path, batchSize=args.detbatch).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()

    # Load detection loader
    print('Beginning people detection')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    # Data writer
    writer = DataWriter(args.save_video, '', cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    # Load CNN model
    image_model = CNN_model(class_num, CNN_pre_train, CNN_pre_train_path, CNN_path)

    for i in range(data_loader.length()):
        if i == 286:
            a = 1
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()

            # Pose Estimation
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm).cpu().data

            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            key_point = writer.pred

            if len(writer.img) > 0 and len(key_point) > 0:
                pos = image_model.predict(writer.img)
                pred = CNN_class[pos[0].tolist().index(max(pos[0].tolist()))]
                coord = [key_point[idx] for idx in nece_point]
                score = max(pos[0].tolist())
                pa = PostureAnalyser(coord)

                if pred == "triangle":
                    res, point = pa.analyse_triangle()
                elif pred == "boat":
                    res, point = pa.analyse_boat()
                elif pred == "tree":
                   res, point = pa.analyse_tree()
                elif pred == "chair":
                    res, point = pa.analyse_chair()
                elif pred == "cow" or pred == "camel":
                    res, point = [], -1
                else:
                    raise ValueError("Wrong prediction")

                point = point if type(point) is float or type(point) is int else point.tolist()
                print("Here is the {} frame".format(i))
                print("Prediction is {}".format(pred))
                if point != -1:
                    # point = round(point * score, 2)
                    print("The score is {}".format(point))
                    for item in res:
                        print(item)
                else:
                    print("Action {} has no standard posture".format(pred))

                print("\n")

    print("total frame is {}".format(i))


if __name__ == "__main__":
    run(video_path)
