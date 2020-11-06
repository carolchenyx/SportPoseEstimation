from config import config
import cv2
import time
from app_class.exercise_src.processor import ImageProcessor
from utils.utils import Utils

IP = ImageProcessor()

# video_path = config.push_up_video_path
body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
nece_point = [5, 7, 9, 13, 15, 6, 8, 10, 14, 16]


def run():
    video_path = config.push_up_video_path
    # video_path = 0
    frm_cnt = 0
    cap = cv2.VideoCapture(video_path)
    count = 0
    up_flag = 0
    push_flag = 0
    count_usd = 0
    count_push = 0
    count_up = 0
    begin_time = 0
    end_time = 0
    angle = 0

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
                right_angle = Utils.get_angle(coord[6], coord[7], coord[5])
                left_angle_1 = Utils.get_angle(coord[3], coord[2], coord[4])
                right_angle_1 = Utils.get_angle(coord[8], coord[7], coord[9])
                if key_point[0].tolist()[0] < 450:
                    angle = left_angle
                    angle_1 = left_angle_1
                else:
                    angle = right_angle
                    angle_1 = right_angle_1

                # a = index.LoadingBar.add_values(angle)

                if angle_1 < 140:
                    count_usd += 1
                    if count_usd > 5:
                        cv2.putText(img, "The posture doesn't satisfy standard", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        pass
                else:
                    count_usd = 0
                    if angle > 130:
                        count_push = 0 if count_push == 0 else count_push - 1
                        if count_up < 5:
                            count_up += 1
                        else:
                            count_up = 0
                            up_flag = 1
                            if push_flag == 1:
                                count += 1
                                if end_time == 0:
                                    begin_time = first_time
                                else:
                                    begin_time = end_time

                                end_time = time.time()
                                push_flag = 0
                            else:
                                pass
                    else:
                        if up_flag == 1:
                            count_up = 0 if count_up == 0 else count_up - 1
                            if count_push > 4:
                                up_flag = 0
                                push_flag = 1
                                count_push = 0
                            else:
                                count_push += 1
                        else:
                            pass

                    cost = int(end_time - float(begin_time))
                    if cost == 0:
                        spd = 0
                    else:
                        spd = '{:.2f}'.format(60/cost)
                    cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    cv2.putText(img,"The angle of arm is {} degree".format(angle),(100,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2)
                    cv2.putText(img, "The last one spends {} seconds".format(cost), (100, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (100, 0, 0), 2)
                    cv2.putText(img, "Now speed is {} times per minute".format(spd), (100, 500),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (100, 0, 0), 2)
                    #cv2.rectangle(img, (750, 0), (950, 150), (0, 255, 0))
                    #cv2.line(img, (850, 150), (950, 66), (0, 255, 0), 3)
                    #cv2.line(img, (850, 150), (750, 66), (0, 255, 0), 3)
                    print("The angle is {} degree\n".format(angle))
                    cv2.imshow("result", img)
                    cv2.waitKey(2)

            else:
                cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.imshow("result", img)
                cv2.waitKey(2)
        else:
            break
    print("The final count is {}".format(count))
    return angle

if __name__ == "__main__":
    run()
