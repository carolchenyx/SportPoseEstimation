from config import config
from estimator.opt import opt
from estimator.dataloader import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from SPPE.src.main_fast_inference import *
from app_class.golf_src.models.CNN_model import CNN_model
import sys
import cv2
from app_class.golf_src.analyzer import PostureAnalyser
from app_class.golf_src.locator import Locator
import socket
import os

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "127.0.0.1"
UDP_PORT = 5066

os.makedirs("log/time_log", exist_ok=True)
file = open("log/time_log/log_half.txt", "w")
args = opt
args.dataset = 'coco'
mode = args.mode
batchSize = args.posebatch

video_path = config.golf_video_path
CNN_path = config.CNN_golf_model
CNN_pre_train = config.CNN_golf_pre_train_model
CNN_class = config.CNN_golf_classes
CNN_pre_train_path = "models/pre_train_model/{}.pth".format(CNN_pre_train)
class_num = len(CNN_class)

args.sp = True
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

nece_point = [5, 7, 9, 15, 6, 8, 10, 16]
reflect = {"Backswing": "b", "Standing": "s", "FollowThrough": "f"}


def run(video_path):
    # Load input video
    data_loader = VideoLoader(video_path, batchSize=args.detbatch).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()
    width, height = frameSize[0], frameSize[1]

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
    pose_model.eval()

    # Data writer
    writer = DataWriter(args.save_video, '', cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    # Load CNN model
    image_model = CNN_model(class_num, CNN_pre_train, CNN_pre_train_path, CNN_path)

    for i in range(data_loader.length()):
        with torch.no_grad():
            begin_time = time.time()
            time_str = time.time()

            file.write("Here is the iteration {}\n".format(str(i)))
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            file.write("Time for detector is {}\n".format(time.time() - time_str))

            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            # Pose Estimation
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []

            time_str = time.time()
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm).cpu().data
            file.write("Time for extractor is {}\n".format(time.time() - time_str))

            time_str = time.time()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            key_point = writer.pred
            file.write("Time for data writer is {}\n".format(time.time() - time_str))

            if len(writer.img) > 0 and len(key_point) > 0:
                time_str = time.time()
                pos = image_model.predict(writer.img)
                file.write("Time for classifier is {}\n".format(time.time() - time_str))

                time_str = time.time()
                pred = CNN_class[pos[0].tolist().index(max(pos[0].tolist()))]
                score = max(pos[0].tolist())
                coord = [key_point[idx] for idx in nece_point]

                if Locator.detect_user(coord[3][1], key_point[0][1], height):
                    pa = PostureAnalyser(coord)
                    if pred == "Backswing":
                        res, point, signal = pa.analyse_backswing()
                    elif pred == "Standing":
                        res, point, signal = pa.analyse_standing()
                    elif pred == "FollowThrough":
                        res, point, signal = pa.analyse_final()
                    else:
                        raise ValueError("Wrong prediction")

                    point = point if type(point) is float or type(point) is int else point.tolist()
                    point = round(score * point, 2)
                    unity_string = reflect[pred] + signal

                    print("Prediction is {}".format(pred))
                    print("The score is {}".format(point))
                    for item in res:
                        print(item)
                    print("\n")

                    sock.sendto((unity_string).encode(), (UDP_IP, UDP_PORT))
                else:
                    print("detecting user...")
                file.write("Time for analyzer is {}\n".format(time.time() - time_str))
                file.write("Time for the whole iteration is {}\n\n".format(time.time() - begin_time))


if __name__ == "__main__":
    run(video_path)
