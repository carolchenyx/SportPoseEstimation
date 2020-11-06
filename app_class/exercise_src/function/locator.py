
class Locator(object):
    def __init__(self, base=(0.5, 0)):
        self.base_location = base
        self.main_human_idx = 0

    def check_by_length(self, humans):
        human_len = [abs(humans[num][0][0] - humans[num][-1][0]) for num in range(len(humans))]
        self.main_human_idx = human_len.index(max(human_len))

    def locate_user(self, kps):
        humans = [kps[i]["keypoints"].tolist() for i in range(len(kps))]
        self.check_by_length(humans)
        return kps[self.main_human_idx]
