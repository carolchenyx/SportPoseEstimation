from config import config
from app_class.golf_src.models.CNN_model import CNN_model
from app_class.golf_src.analyzer import PostureAnalyser
from estimator.opt import opt
import socket
from estimator.dataloader_webcam import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from SPPE.src.main_fast_inference import *
import os
import sys
from tqdm import tqdm
from app_class.golf_src.locator import Locator
import cv2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "127.0.0.1"
UDP_PORT = 5066


file = open("log_webcam.txt", "w")
args = opt
args.dataset = 'coco'
mode = args.mode
batchSize = args.posebatch

webcam = config.golf_webcam_num
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


def loop():
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":
    mode = args.mode

    # Load input video
    data_loader = WebcamLoader(webcam).start()
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
    pose_model.cuda()
    pose_model.eval()

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(webcam) + '.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()
    image_model = CNN_model(class_num, CNN_pre_train, CNN_pre_train_path, CNN_path)

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())

    for i in im_names_desc:
        try:
            begin_time = time.time()
            time_str = time.time()

            file.write("Here is the iteration {}\n".format(str(i)))
            print("Here is the {} frame".format(i))
            with torch.no_grad():
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
                hm = torch.cat(hm)

                hm = hm.cpu().data
                file.write("Time for extractor is {}\n".format(time.time() - time_str))

                writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                key_point = writer.pred
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
                        unity_string = reflect[pred] + signal

                        file.write("Time for analyzer is {}\n".format(time.time() - time_str))
                        file.write("Time for the whole iteration is {}\n\n".format(time.time() - begin_time))

                        print("Prediction is {}".format(pred))
                        print("The score is {}".format(point))
                        for item in res:
                            print(item)
                        print("\n")

                        sock.sendto((unity_string).encode(), (UDP_IP, UDP_PORT))
                    else:
                        print("detecting user...")
                else:
                    pass
                file.write("\n")

        except KeyboardInterrupt:
            break

    print('===========================> Finish Model Running.')
    while (writer.running()):
        pass
    writer.stop()
    final_result = writer.results()

