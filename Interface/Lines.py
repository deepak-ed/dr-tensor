from PyQt5.QtWidgets import QGraphicsItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsView, QGraphicsScene, \
    QHBoxLayout, QGraphicsPixmapItem, QVBoxLayout, QGraphicsRectItem
from PyQt5.QtCore import Qt, QPoint, QLineF, QRectF
from PyQt5.QtGui import QColor, QPen, QImage, QPixmap, QPainter
from PyQt5 import QtWidgets
import math
import numpy as np
from Backend.network.femur_detector import runner


class Tab(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.pic = None
        self.points = None
        self.all_lines = None

    def getScene(self):
        return self.scene

    def openImage(self, fname):

        image_qt = QImage(fname)
        pic_item = QGraphicsPixmapItem()
        pic_item.setPixmap(QPixmap.fromImage(image_qt).scaled(self.geometry().width(),  self.geometry().height()))
        pic_item.setFlag(QGraphicsItem.ItemIsMovable)
        pic_item.setFlag(QGraphicsItem.ItemIsSelectable)

        self.pic = pic_item
        self.scene.addItem(pic_item)
        print("start setup")
        self.setup()

    def setup(self):
        self.points = self.get_predicted_points()
        if self.all_lines is not None:
            for line in self.all_lines:
                self.scene.removeItem(line)
            self.all_lines = list()
        else:
            self.all_lines = list()

        for key, value in self.points.items():  # key - Left/Right
            for key2, value2 in value.items():  # key - Line(Name)
                new_Line = LineItem(value2[0], value2[1], value2[2], value2[3], key, key2, max(self.pic.pixmap().height(), self.pic.pixmap().width()))
                new_Line.setParentItem(self.pic)
                self.all_lines.append(new_Line)

        for line in self.all_lines:
            self.scene.addItem(line)
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)


class Tab2(QtWidgets.QWidget):
    def __init__(self, parent=None):
        # for zoom: if both have left: zoom in both images, if only one left femur: maximize this view
        super().__init__(parent)
        self.layout = QHBoxLayout(self)

        self.views = list([QGraphicsView(), QGraphicsView()])

        self.scene = list([QGraphicsScene(), QGraphicsScene()])

        self.views[0].setScene(self.scene[0])
        self.views[1].setScene(self.scene[1])

        self.pic = list()
        self.points = None
        self.all_lines = None

        self.angle_left = "N/A"
        self.angle_right = "N/A"

    def getScene(self, i=0):
        return self.scene[i]

    def openImage(self, fname, runner):
        for i in range(2):
            image_qt = QImage(fname[i])

            pic_item = QGraphicsPixmapItem()
            pic_item.setPixmap(QPixmap.fromImage(image_qt)) #.scaled(self.views[i].geometry().width()
            #pic_item.setFlag(QGraphicsItem.ItemIsMovable)
            #pic_item.setFlag(QGraphicsItem.ItemIsSelectable)

            self.pic.append(pic_item)
            self.scene[i].addItem(pic_item)
        self.setup(fname, runner)

    def reset(self):
        if self.all_lines is not None:
            for i in range(2):
                for line in self.all_lines[i]:
                    self.scene[i].removeItem(line)
                self.all_lines[i] = list()
        else:
            self.all_lines = list([[], []])

        for i in range(2):
            for key, value in self.points[i].items():  # key - Left/Right
                for key2, value2 in value.items():  # key - Line(Name)
                    if value2 is None:
                        continue
                    if "centerline" in key2 or "Centerline" in key2:
                        new_Line = LineItem(int(value2[0]), int(value2[1]), int(value2[2]), int(value2[3]), key, key2, max(self.pic[i].pixmap().height(), self.pic[i].pixmap().width()))
                        new_Line.setParentItem(self.pic[i])
                        self.all_lines[i].append(new_Line)

            for line in self.all_lines[i]:
                line.show()
                self.scene[i].addItem(line)

        self.views[0].fitInView(self.pic[0], Qt.KeepAspectRatio)
        self.views[1].fitInView(self.pic[1], Qt.KeepAspectRatio)


    def setup(self, fname, detec):
        first_open = False
        if self.points is None:
            first_open = True
            self.points = list([self.get_predicted_points(fname[0], detec), self.get_predicted_points(fname[1], detec)])

        if self.all_lines is not None:
            for i in range(2):
                for line in self.all_lines[i]:
                    self.scene[i].removeItem(line)
                self.all_lines[i] = list()
        else:
            self.all_lines = list([[], []])
        count_right = [0, 0]
        for i in range(2):
            for key, value in self.points[i].items():  # key - Left/Right
                for key2, value2 in value.items():  # key - Line(Name)
                    if value2 is None:
                        continue
                    if key == "Right":
                        count_right[i] += 1
                    if "centerline" in key2 or "Centerline" in key2:
                        new_Line = LineItem(int(value2[0]), int(value2[1]), int(value2[2]), int(value2[3]), key, key2, max(self.pic[i].pixmap().height(), self.pic[i].pixmap().width()))
                        new_Line.setParentItem(self.pic[i])
                        self.all_lines[i].append(new_Line)
            if first_open:
                index = np.argmax(count_right)
                self.layout.addWidget(self.views[index])
                self.layout.setStretchFactor(self.views[index], 1)

                index = 0 if index == 1 else 1

                self.layout.addWidget(self.views[index])
                self.layout.setStretchFactor(self.views[index], 1)
            for line in self.all_lines[i]:
                self.scene[i].addItem(line)
        self.views[0].fitInView(self.pic[0], Qt.KeepAspectRatio)
        self.views[1].fitInView(self.pic[1], Qt.KeepAspectRatio)

    def get_predicted_points(self, fname, detec):
        pred_points = runner(detec, fname)
        return pred_points

    # TODO resize funktioniert nicht, warum?
    def maximize_right(self):
        self.layout.setStretch(0, 5)
        self.layout.setStretch(1, 1)

        self.views[0].fitInView(self.pic[0], Qt.KeepAspectRatio)
        self.views[1].fitInView(self.pic[1], Qt.KeepAspectRatio)

    def zoom_out(self):
        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 1)
        self.layout.update()

        self.views[0].fitInView(self.pic[0], Qt.KeepAspectRatio)
        self.views[1].fitInView(self.pic[1], Qt.KeepAspectRatio)

    def maximize_left(self):
        self.layout.setStretch(1, 5)
        self.layout.setStretch(0, 1)

class LineItem(QGraphicsLineItem):
    def __init__(self, y1, x1, y2, x2, position: str,
                 description: str, max_size):  # position: Left/Right, description: Line Name(for example Shaft Centerline)
        super().__init__()
        # calculate width/size of circle and lines
        circle_size = 2 * round((max_size/83) / 2)
        line_width = math.ceil(circle_size / 5)

        self.setLine(x1, y1, x2, y2)
        if position == "Left":
            color = Qt.blue
        elif position == "Right":
            color = Qt.green
        else:
            color = Qt.red

        self.setPen(QPen(color, line_width))
        #self.setFlag(QGraphicsItem.ItemIsMovable)
        #self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.endpoints = [Endpoint(self, circle_size, color) for _ in range(2)]
        self.endpoints[0].setParentItem(self)
        self.endpoints[1].setParentItem(self)

        self.endpoints[0].setPos(QPoint(x1, y1))
        self.endpoints[1].setPos(QPoint(x2, y2))

        self.position = position
        self.description = description

    def updateLine(self):
        self.setLine(QLineF(self.endpoints[0].pos(), self.endpoints[1].pos()))


class Endpoint(QGraphicsEllipseItem):
    def __init__(self, parent, circle_size, color):
        super().__init__(parent)
        self.color = color
        self.hover_col = QColor(223, 164, 69)

        self.setRect(-circle_size/2, -circle_size/2, circle_size, circle_size) #-15, 30
        self.setBrush(self.color)

        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.setBrush(self.hover_col)
        self.update()

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(self.color)

    def itemChange(self, change, value):
        if change == self.ItemPositionHasChanged:
            self.parentItem().updateLine()
        return super().itemChange(change, value)


def onSegment(p, q, r):
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False


def orientation(p, q, r):
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Collinear orientation
        return 0


def doIntersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


def lineLineIntersection(A, B, C, D):
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])

    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])

    determinant = a1 * b2 - a2 * b1

    if (determinant == 0):
        return False
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return [x, y]
