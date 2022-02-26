import atexit
from dataclasses import dataclass, field
from itertools import combinations
from random import random
from typing import AnyStr, List, SupportsInt, Tuple, Union, TypeVar, Callable

import cv2 as cv
import numpy as np

from ocr import get_num

Number = TypeVar('Number', float, int)


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"Point({self.x:.2f},{self.y:.2f})"

    def __iter__(self):
        return iter([self.x, self.y])


class Line:
    def __init__(self, rho, theta, max_num=1e5):
        self.rho = rho
        self.theta = theta
        self.max_num = max_num or 1e5

    def __repr__(self):
        return f"Line<rho={self.rho:.2f},theta={self.theta:.2f}>"

    def to_line(self) -> Tuple[float, float, float]:
        """

        :return: x*cos(theta)+y*sin(theta)=rho => x*A+y*B=C
        """
        return np.cos(self.theta), np.sin(self.theta), self.rho

    def to_2point(
            self, obj=False
    ) -> Union[
        Tuple[Point, Point], Tuple[Number, Number, Number, Number]
    ]:
        A, B, C = self.to_line()
        x0 = A * C
        y0 = B * C
        x1 = int(x0 + self.max_num * (-B))
        y1 = int(y0 + self.max_num * A)
        x2 = int(x0 - self.max_num * (-B))
        y2 = int(y0 - self.max_num * A)
        if obj:
            return Point(x1, y1), Point(x2, y2)
        else:
            return x1, y1, x2, y2

    @staticmethod
    def createFromPoints(x1, y1, x2, y2):
        raise NotImplementedError()

    @staticmethod
    def createFromHoughSpace(rho, theta, max_num=None):
        return Line(rho, theta, max_num=max_num)

    def __and__(self, other) -> Union[Point, None]:
        return self.intersection(other)

    def __floordiv__(self, other):
        diff = np.fabs(self.theta - other.theta)
        if diff < np.pi * 1e-1:
            return True
        else:
            return False

    def __sub__(self, other):
        if self // other:
            return np.fabs(self.rho - other.rho)
        else:
            raise False

    def intersection(self, other) -> Union[Point, None]:
        l1 = self.to_line()
        l2 = other.to_line()
        # A*[x,y] = B
        A = np.array([l1[0:2], l2[0:2]])
        B = np.array([l1[2:], l2[2:]])
        try:
            x, y = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return None
        x = x[0] if x else None
        y = y[0] if y else None
        if not x or not y or x > self.max_num or y > self.max_num:
            return None
        return Point(x, y)

    def __eq__(self, other):
        return (
                np.fabs(self.rho - other.rho) < 1e-2
                and np.fabs(self.theta - other.theta) < 1e-2
        )


@dataclass()
class ImgField:
    data: np.ndarray
    x_center: int
    y_center: int
    index: int
    number: Union[Number, None] = field(init=False)

    def __repr__(self):
        return f"ImgField<index={self.index},center_pos=({self.x_center},{self.y_center}),WxH={self.data.shape[1]}x{self.data.shape[0]}>"

    def toImgBytes(self) -> bytes:
        _, img = cv.imencode('.jpg', self.data)
        return img.tobytes()


@dataclass()
class CropInfo:
    img: np.ndarray = field(default=None, init=False)
    x_slice: slice
    y_slice: slice
    x_center: int
    y_center: int

    def __repr__(self):
        return f"Crop<{self.x_slice.start}:{self.x_slice.stop},{self.y_slice.start}:{self.y_slice.stop}>"

    def pt1(self):
        return self.x_slice.start, self.y_slice.start

    def pt2(self):
        return self.x_slice.stop, self.y_slice.stop

    def center(self):
        return self.x_center, self.y_center


def resize(img: np.ndarray, ratio=0.5, dsize=None, interpolation=cv.INTER_LINEAR) -> np.ndarray:
    if dsize:
        ds = dsize
    elif ratio:
        ds = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    else:
        raise ValueError("Must provide ratio or dsize!")
    return cv.resize(img, dsize=ds, interpolation=interpolation)


def show_img(img: np.ndarray, win_name=None, ratio=0.5, wait_key=True) -> AnyStr:
    height, width, _ = img.shape
    win_name = str(win_name or random())
    cv.namedWindow(win_name, cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL | cv.WINDOW_GUI_EXPANDED)
    if ratio:
        pass
        # img = resize(img, ratio=ratio)
    cv.imshow(win_name, img)
    cv.resizeWindow(win_name, int(width * ratio), int(height * ratio))
    if wait_key:
        atexit._clear()
        atexit.register(cv.waitKey)
    return win_name


_B = _G = _R = TypeVar('Number', int, float)


def draw_line(img: np.ndarray,
              lines: List[Line],
              color: Tuple[_B, _G, _R] = None,
              thickness: SupportsInt = None, copy=False) -> np.ndarray:
    if copy:
        img = img.copy()
    color = color or (0, 0, 255)
    thickness = thickness or 2
    for i in lines:
        t = i.to_2point(obj=False)
        cv.line(img, t[0:2], t[2:4], color=color, thickness=thickness)
    return img


def line_detection(image: np.ndarray) -> List[Line]:
    if image.shape[2] != 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # apertureSize是sobel算子大小，只能为1,3,5，7
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines: np.ndarray = cv.HoughLines(edges, 0.1, np.pi / 180, 250)
    return list(
        map(lambda x: Line.createFromHoughSpace(*x[0], max_num=max(image.shape) * 1.5),
            lines.tolist()))


def _split2parallel_line(lines: List[Line], d_theta_rad=np.pi / 180 * 1) -> List[List[Line]]:
    """
    将互相平行的直线放入同一个列表中
    :param lines: 直线列表
    :param d_theta_rad: 判定平行时，直线角度的允许误差(弧度)
    :return:
    """
    parallel_line_list = [[]]
    for i in sorted(lines, key=lambda x: x.theta):
        tmp = parallel_line_list[-1]
        if tmp:
            if abs(tmp[0].theta - i.theta) < d_theta_rad:
                tmp.append(i)
            else:
                parallel_line_list.append([i])
        else:
            tmp.append(i)
    return parallel_line_list


def clean_lines(lines: List[Line], d_rho=10, d_theta=np.pi / 180 * 1) -> List[Line]:
    # 将平行的直线放在同一个list中
    parallel_line_list = _split2parallel_line(lines, d_theta_rad=d_theta)
    # 将距离间隔过小的直线放在同一个list中
    near_line_list = []
    for l in parallel_line_list:
        # 避免将不同的平行线放在同一个list中
        near_line_list.append([])
        for i in sorted(l, key=lambda x: x.rho):
            s = near_line_list[-1]
            if s:
                if abs(s[-1].rho - i.rho) < d_rho:
                    s.append(i)
                else:
                    near_line_list.append([i])
            else:
                s.append(i)
    cleaned_line_list = []
    for i in near_line_list:
        rho = sum([j.rho for j in i]) / len(i)
        theta = sum([j.theta for j in i]) / len(i)
        cleaned_line_list.append(Line(rho, theta))
    final_line_list = []
    for i in _split2parallel_line(cleaned_line_list, d_theta_rad=d_theta):
        # 选出其中相隔距离的方差最小的十条直线
        target = min(combinations(i, 10),
                     key=lambda array:
                     np.diff(sorted(map(lambda x: x.rho, array))).var()
                     )
        final_line_list += target
    return final_line_list


def check_lines(lines: List[Line]) -> bool:
    pl = _split2parallel_line(lines)
    if len(pl) != 2 or len(lines) != 20:
        return False
    else:
        return True


def get_crop_info(lines: List[Line], shrink_border_percent=0.05) -> List[List[CropInfo]]:
    m, n = _split2parallel_line(lines)
    m = sorted(m, key=lambda x: x.rho)
    n = sorted(n, key=lambda x: x.rho)
    ret_list = []
    for i in range(1, len(m)):
        i1, i2 = m[i - 1], m[i]
        tmp = []
        for j in range(1, len(n)):
            j1, j2 = n[j - 1], n[j]
            p1: Point = i1 & j1
            p2: Point = i2 & j2
            shrink_border_width = min(p2.x - p1.x, p2.y - p1.y) * shrink_border_percent
            tmp.append(
                CropInfo(x_slice=slice(int(p1.x + shrink_border_width), int(p2.x - shrink_border_width)),
                         y_slice=slice(int(p1.y + shrink_border_width), int(p2.y - shrink_border_width)),
                         x_center=int((p1.x + p2.x) / 2),
                         y_center=int((p1.y + p2.y) / 2)
                         ))
        ret_list.append(tmp)
    return ret_list


def crop(img: np.ndarray,
         crop_info_list: List[List[CropInfo]]) -> List[ImgField]:
    img = img.copy()
    ret = []
    index = 0
    for j in crop_info_list:
        for i in j:
            ret.append(ImgField(
                data=img[i.y_slice, i.x_slice],
                x_center=i.x_center,
                y_center=i.y_center, index=index))
            index += 1
    return ret


def draw_border_info(img: np.ndarray, crop_info: List[List[CropInfo]], copy=False) -> np.ndarray:
    if copy:
        img = img.copy()
    count = 0
    for i in crop_info:
        for j in i:
            cv.putText(img, str(count), j.center(), None, fontScale=0.5, color=(0, 255, 0),
                       thickness=2)
            cv.rectangle(img, j.pt1(), j.pt2(), color=(0, 0, 255), thickness=2)
            count += 1
    return img


def padding2square(img: np.ndarray, target_square_length=None) -> np.ndarray:
    height, width, _ = img.shape
    bottom, right = 0, 0
    if height > width:
        right = height - width
    else:
        bottom = width - height

    ret = cv.copyMakeBorder(img, 0, bottom, 0, right, cv.BORDER_CONSTANT, (255, 255, 255))
    if target_square_length and max(height, width) != target_square_length:
        resize(ret, dsize=(target_square_length, target_square_length))
    return ret


def recognize_puzzle(img: Union[np.ndarray], ocr_func: Callable[[bytes], int] = get_num) -> List[ImgField]:
    lines_obj = line_detection(img)
    lines_obj_cleaned = clean_lines(lines_obj)
    crop_info = get_crop_info(lines_obj_cleaned)
    img_field_list = crop(img, crop_info)
    for i in img_field_list:
        i.number = ocr_func(i.toImgBytes())
    return img_field_list


if __name__ == '__main__':
    i = cv.imread('../web_sudoku.png')
    ifs = recognize_puzzle(i, get_num)
    sudoku = []
    for i in ifs:
        sudoku.append(i.number)
    print(sudoku)
