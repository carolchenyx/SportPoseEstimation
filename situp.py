from config import config
import cv2
import time
from app_class.exercise_src.processor import ImageProcessor
from utils.utils import Utils

IP = ImageProcessor()

video_path = config.sit_up_video_path
body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
nece_point = [5, 7, 9, 11, 13, 6, 8, 10, 12, 14]


def run(video_path):
    frm_cnt = 0
    cap = cv2.VideoCapture(video_path)
    count = 0
    sit_flag = 0
    down_flag = 0
    count_down = 0
    count_sit = 0
    begin_time = 0
    end_time = 0
    count_usd = 0

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
                left_angle = Utils.get_angle(coord[3], coord[0], coord[4])
                right_angle = Utils.get_angle(coord[8], coord[5], coord[9])
                left_angle_1 = Utils.get_angle(coord[1], coord[0], coord[2])
                right_angle_1 = Utils.get_angle(coord[6], coord[5], coord[7])

                #if key_point[0].tolist()[0] > 450:
                    #angle = left_angle
                    #angle_1 = left_angle_1
                #else:
                    #angle = right_angle
                    #angle_1 = right_angle_1
                if left_angle_1 > 110 or right_angle_1 > 110:
                    count_usd += 1
                    if count_usd > 5:
                       cv2.putText(img, "The posture doesn't satisfy standard", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        pass
                else:
                    count_usd = 0
                    if left_angle > 110 or right_angle > 110:
                        count_sit = 0 if count_sit == 0 else count_sit - 1
                        if count_down < 5:
                            count_down += 1
                        else:
                            count_down = 0
                            down_flag = 1
                        #if sit_flag == 1:
                            #count += 1
                            #sit_flag = 0
                       # else:
                           # pass

                    else:
                        if down_flag == 1:
                            count_down = 0 if count_down == 0 else count_down - 1
                            if count_sit > 4:
                                down_flag = 0
                                #sit_flag = 1
                                count += 1
                                if end_time == 0:
                                    begin_time = first_time
                                else:
                                    begin_time = end_time

                                end_time = time.time()
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
                cv2.putText(img, "The angle of hip is {} degree".format(right_angle), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
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
