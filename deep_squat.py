from config import config
import cv2
import time
from app_class.exercise_src.processor import ImageProcessor
from utils.utils import Utils

IP = ImageProcessor()

video_path = config.deep_squat_video_path
body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
nece_point = [11, 13, 15, 12, 14, 16]


def run(video_path):
    frm_cnt = 0
    cap = cv2.VideoCapture(video_path)
    count = 0
    stand_flag = 0
    sit_flag = 0
    count_stand = 0
    count_sit = 0
    begin_time = 0
    end_time = 0

    while True:
        frm_cnt += 1
        print("cnt {}".format(count))
        print("frame {}".format(frm_cnt))
        ret, frame = cap.read()
        if ret:
            key_point, img, _ = IP.process_img(frame)
            if frm_cnt == 1:
                first_time = time.time()
            else:
                pass
            if len(img) > 0 and len(key_point) > 0:
                coord = [key_point[idx] for idx in nece_point]
                left_angle = Utils.get_angle(coord[1], coord[0], coord[2])
                right_angle = Utils.get_angle(coord[4], coord[3], coord[5])
                if right_angle > 150 and left_angle > 150:
                    count_sit = 0 if count_sit == 0 else count_sit - 1
                    if count_stand < 5:
                        count_stand += 1
                    else:
                        count_stand = 0
                        stand_flag = 1
                        if sit_flag == 1:
                            count += 1
                            if end_time == 0:
                                begin_time = first_time
                            else:
                                begin_time = end_time

                            end_time = time.time()
                            sit_flag = 0
                        else:
                            pass
                else:
                    if stand_flag == 1:
                        count_stand = 0 if count_stand == 0 else count_stand - 1
                        if count_sit > 4:
                            stand_flag = 0
                            sit_flag = 1
                            count_sit = 0
                        else:
                            count_sit += 1
                    else:
                        pass

                cost = int(end_time - float(begin_time))
                if cost == 0:
                    spd = 0
                else:
                    spd = '{:.2f}'.format(60 / cost)
                cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.putText(img, "The angle of leg is {} degree".format(right_angle), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (100, 0, 0), 2)
                cv2.putText(img, "The last one spends {} seconds".format(cost), (100, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (100, 0, 0), 2)
                cv2.putText(img, "Now speed is {} times per minute".format(spd), (100, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (100, 0, 0), 2)
                print("The angle is {} degree\n".format(right_angle))
                cv2.imshow("result", img)
                cv2.waitKey(2)

            else:
                cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.imshow("result", img)
                cv2.waitKey(2)
        else:
            break
    print("The final count is {}".format(count))

if __name__ == "__main__":
        run(video_path)