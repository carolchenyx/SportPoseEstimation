from config import config
import cv2
from app_class.exercise_src.processor import ImageProcessor
from utils.utils import Utils

IP = ImageProcessor()

video_path = config.push_up_video_path
body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
nece_point = [5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]


def run(video_path):
    frm_cnt = 0
    cap = cv2.VideoCapture(video_path)
    count = 0
    up_flag = 0
    count_push = 0
    count_up = 0
    count_unstandard = 0

    while True:
        frm_cnt += 1
        print("cnt {}".format(count))
        print("frame {}".format(frm_cnt))
        ret, frame = cap.read()

        if ret:
            key_point, img, _ = IP.process_img(frame)
            if len(img) > 0 and len(key_point) > 0:
                coord = [key_point[idx] for idx in nece_point]
                left_angle = Utils.get_angle(coord[1], coord[0], coord[2])
                right_angle = Utils.get_angle(coord[7], coord[6], coord[8])
                left_leg = Utils.get_angle(coord[4], coord[3], coord[5])
                right_leg = Utils.get_angle(coord[10], coord[9], coord[11])
                if key_point[0][0] > 450:
                    angle = right_angle
                else:
                    angle = left_angle

                if count_unstandard < 4:
                    if left_leg > 165 and right_leg > 165:
                        if angle > 130:
                            count_push = 0 if count_push == 0 else count_push - 1
                            # if count_push == 0:
                            #     pass
                            # else:
                            #     count_push -= 1
                            if count_up < 5:
                                count_up += 1
                            else:
                                count_up = 0
                                up_flag = 1
                        else:
                            if up_flag == 1:
                                count_up = 0 if count_up == 0 else count_up - 1
                                if count_push > 4:
                                    count += 1
                                    up_flag = 0
                                    count_push = 0
                                else:
                                    count_push += 1
                            else:
                                pass
                    else:
                        count_unstandard +=1
                else:
                    cv2.putText(img, "Pose doesn't satisfy demand", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    count_unstandard -= 1



                cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
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

