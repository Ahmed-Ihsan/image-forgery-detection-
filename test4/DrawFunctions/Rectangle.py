import cv2
import numpy as np
from scipy.stats import stats

from DrawFunctions.AbstractShape import AbstractShape


class DrawRectangle(AbstractShape):

    image = None
    keypoints1 = None
    keypoints2 = None
    color = None
    cRectangle = None

    def __init__(self, image, keypoints1, keypoints2, color, count_rectangle):
        self.image = image
        self.keypoints1 = keypoints1
        self.keypoints2 = keypoints2
        self.color = color
        self.cRectangle = count_rectangle  # counts of rectangle
        self.draw()

    def draw(self, **kwargs):
        new_image = self.image.copy()

        if self.cRectangle == 0:
            k1x, k2x = np.max(self.keypoints1, axis=0), np.max(self.keypoints2, axis=0)
            k1n, k2n = np.min(self.keypoints1, axis=0), np.min(self.keypoints2, axis=0)
            cv2.rectangle(new_image, (int(k2x[0]) + 10, int(k2n[1]) - 10), (int(k2n[0]) - 10, int(k2x[1]) + 10), self.color, 3)
            cv2.rectangle(new_image, (int(k1x[0]) + 10, int(k1n[1]) - 10), (int(k1n[0]) - 10, int(k1x[1]) + 10), self.color, 3)
            self.image = new_image
        # elif self.cRectangle == 3:
        #     point_list, z = np.zeros(len(self.keypoints1)), 0
        #     z2, z3, z4 = np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]])
        #     for k1, k2 in zip(self.keypoints1, self.keypoints2):
        #         if len(self.keypoints1) > 1:
        #             p = (k1[0] - k2[0]) / (k1[1] - k2[1])
        #             point_list[z] = int(p)
        #             z = z + 1
        #     for k1, k2 in zip(self.keypoints1, self.keypoints2):
        #         if len(self.keypoints1) > 1:
        #             p = (k1[0] - k2[0]) / (k1[1] - k2[1])
        #             p = int(p)
        #             if p == max(point_list):
        #                 newrow = [k1[0], k1[1]]
        #                 z2 = np.vstack([z2, newrow])
        #             elif p < 0:
        #                 newrow = [k1[0], k1[1]]
        #                 z3 = np.vstack([z3, newrow])
        #                 newrow = [k2[0], k2[1]]
        #                 z4 = np.vstack([z4, newrow])
        #
        #     k1x, k11x, k2x = np.max(z3, axis=0), np.max(z2, axis=0), np.max(z4, axis=0)
        #     z2[0], z3[0], z4[0] = k11x, k1x, k2x
        #     k11n, k1n, k2n = np.min(z2, axis=0), np.min(z3, axis=0), np.min(z4, axis=0)
        #
        #     cv2.rectangle(new_image, (int(k2x[0]) + 10, int(k2n[1]) - 10), (int(k2n[0]) - 10, int(k2x[1]) + 10), self.color, 3)
        #     cv2.rectangle(new_image, (int(k11x[0]) + 10, int(k11n[1]) - 10), (int(k11n[0]) - 10, int(k11x[1]) + 10), self.color, 3)
        #     cv2.rectangle(new_image, (int(k1x[0]) + 10, int(k1n[1]) - 10), (int(k1n[0]) - 10, int(k1x[1]) + 10), self.color, 3)
        # self.image = new_image
        elif self.cRectangle == 3:
            egimlist, x = np.empty(0), 0
            reclist1, reclist2, reclist3 = np.empty(shape=[0, 2]), np.empty(shape=[0, 2]), np.empty(shape=[0, 2])
            for k1, k2 in zip(self.keypoints1, self.keypoints2):
                if len(self.keypoints1) > 1:
                    egim = (k1[0] - k2[0]) / (k1[1] - k2[1])
                    egim = int(egim)
                    egimlist = np.append(egimlist, [egim])
            mode = stats.mode(egimlist)

            while x != len(egimlist):
                if egimlist[x] == mode[0]:
                    egimlist = np.delete(egimlist, x)
                else:
                    x = x + 1
            mode2 = stats.mode(egimlist)
            for k1, k2 in zip(self.keypoints1, self.keypoints2):
                if len(self.keypoints1) > 1:
                    egim = (k1[0] - k2[0]) / (k1[1] - k2[1])
                    egim = int(egim)
                    if egim == mode[0]:
                        reclist1 = np.append(reclist1, [[k1[0], k1[1]]], axis=0)
                        reclist3 = np.append(reclist3, [[k2[0], k2[1]]], axis=0)
                    elif egim == mode2[0] or mode2[1] * 2 <= mode[1]:
                        reclist2 = np.append(reclist2, [[k1[0], k1[1]]], axis=0)

            k1x, k11x, k2x = np.max(reclist2, axis=0), np.max(reclist1, axis=0), np.max(reclist3, axis=0)
            k11n, k1n, k2n = np.min(reclist1, axis=0), np.min(reclist2, axis=0), np.min(reclist3, axis=0)

            cv2.rectangle(new_image, (int(k2x[0]) + 10, int(k2n[1]) - 10), (int(k2n[0]) - 10, int(k2x[1]) + 10),
                          self.color, 3)
            cv2.rectangle(new_image, (int(k11x[0]) + 10, int(k11n[1]) - 10), (int(k11n[0]) - 10, int(k11x[1]) + 10),
                          self.color, 3)
            cv2.rectangle(new_image, (int(k1x[0]) + 10, int(k1n[1]) - 10), (int(k1n[0]) - 10, int(k1x[1]) + 10),
                          self.color, 3)

        self.image = new_image
