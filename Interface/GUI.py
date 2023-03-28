from datetime import datetime
import functools
import glob
import io
import os
import tempfile
import threading
import time

import numpy as np
from PyQt5.QtWidgets import (QMainWindow,
                             QApplication,
                             QPushButton,
                             QLabel,
                             QFileDialog,
                             QGraphicsScene,
                             QGraphicsView,
                             QGraphicsPixmapItem,
                             QInputDialog,
                             QMessageBox)
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QFont, QIcon
from PyQt5.QtCore import QPoint, Qt, QRectF, pyqtSignal
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import queue
import torch
import operator

from Backend.network.femur_detector import runner, load_model
from Interface.Lines import LineItem, Tab2, doIntersect, lineLineIntersection


class UI(QMainWindow):
    openSignal = pyqtSignal(object)
    saveSignal = pyqtSignal(object)

    def __init__(self, femur_detector_model_path=None):
        super(UI, self).__init__()

        uic.loadUi("static/GUI.ui", self)
        self.setWindowTitle("CCD angle interface")
        self.setWindowIcon(QIcon('static/symbol.PNG'))
        self.detector = load_model(femur_detector_model_path)

        # define all buttons and labels
        self.open_im = self.findChild(QPushButton, "open")
        #self.main_view = self.findChild(QGraphicsView, "mainView")
        self.text_right = self.findChild(QLabel, "text_right")
        self.text_left = self.findChild(QLabel, "text_left")
        self.reset = self.findChild(QPushButton, "reset")
        self.voice_im = self.findChild(QLabel, "voice_signal")
        self.save = self.findChild(QPushButton, "save")
        self.speaker_image = self.findChild(QLabel, "speaker_image")
        self.image_sleep = QPixmap('static/mic1.png')
        self.image_listen = QPixmap("static/mic3.png")

        #logic
        self.open_im.clicked.connect(self.click_open)
        self.zoom_right.clicked.connect(self.click_right)
        self.zoom_left.clicked.connect(self.click_left)
        self.reset.clicked.connect(self.setup_scene)
        self.save.clicked.connect(self.save_scene)

        self.openSignal.connect(lambda: self.click_open(voice=True))
        self.saveSignal.connect(lambda : self.save_scene(voice=True))

        self.first_im = False
        self.pic = None
        self.all_lines = list()
        self.predicted_points = None
        self.angle_left = "N/A"
        self.angle_right = "N/A"
        self.listening_state = False
        self.zoom = 0  # 0 : no zoom, 1: right zoom, -1: left zoom

        self.second_im = False
        self.sec_scene = None
        self.sec_view = None
        self.all_lines_2 = list()
        self.count = 0

        self.main_view = None
        self.main_scene = None
        self.tab = None

        self.save_window = None
        self.show()
        self.clock_time = 5.0

        self.voice = queue.Queue()
        threading.Thread(target=self.voice_control, args=("base", False, False, 600, 0.5, False, False), daemon=True).start()

    def voice_control(self, model, english,verbose, energy, pause,dynamic_energy,save_file):
        temp_dir = tempfile.mkdtemp() if save_file else None
        # there are no english models for large
        if model != "large" and english:
            model = model + ".en"
        audio_model = whisper.load_model(model)
        audio_queue = queue.Queue()
        # result_queue = queue.Queue()
        threading.Thread(target=record_audio,
                         args=(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir), daemon=True).start()
        threading.Thread(target=transcribe_forever,
                         args=(audio_queue, self.voice, audio_model, english, verbose, save_file), daemon=True).start()
        in_save = False
        self.speaker_image.setPixmap(self.image_sleep)
        while True:
            res = self.voice.get()
            print(res)
            res = res.lower()
            if in_save:
                if "ok" in res or "okay" in res:
                    temp = res[9:res.find('ok')]
                    self.save_label.setText(self.save_label.text() + temp)
                    self.saveSignal.emit(True)
                    self.save_label.clear()
                    in_save = False
                    self.save_label.setText("saved screenshot")
                    self.listening_state = False
                    self.speaker_image.setPixmap(self.image_sleep)
                else:
                    self.save_label.setText(self.save_label.text() + res[9:])
            else:
                self.save_label.clear()

            if "activate" in res.lower():
                self.listening_state = True
                self.speaker_image.setPixmap(self.image_listen)
                time_now = time.time()

            if self.listening_state and not in_save:
                if "left" in res.lower():
                    self.click_left()
                    self.listening_state = False
                    self.speaker_image.setPixmap(self.image_sleep)
                elif "right" in res.lower() or "ride" in res.lower():
                    self.click_right()
                    self.listening_state = False
                    self.speaker_image.setPixmap(self.image_sleep)
                elif "out" in res.lower() or "both" in res.lower():
                    if self.zoom == 1:
                        self.click_right()
                        self.listening_state = False
                        self.speaker_image.setPixmap(self.image_sleep)
                    elif self.zoom == -1:
                        self.click_left()
                        self.listening_state = False
                        self.speaker_image.setPixmap(self.image_sleep)
                elif "open" in res:
                    self.openSignal.emit(True)
                    self.listening_state = False
                    self.speaker_image.setPixmap(self.image_sleep)
                elif "save" in res.lower() or "safe" in res.lower():
                    now = datetime.now()
                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    self.save_label.setText(dt_string)
                    in_save = True
                elif "left" not in res.lower() or "right" not in res.lower() or "save" not in res.lower() or "safe" not in res.lower() and not in_save:
                    time_present = time.time()
                    if (time_present-time_now) > self.clock_time:
                        self.listening_state =False
                        self.speaker_image.setPixmap(self.image_sleep)
                        print("stop listening")

    def updateAngle(self, all_lines):
        critical = ["Femoral Shaft Centerline", "Femoral Neck Centerline"]
        for p in ["Right", "Left"]:
            points = list()
            for c in critical:
                for line in all_lines:
                    if line.position == p and c in line.description:
                        points.append([line.endpoints[0].pos().x(), line.endpoints[0].pos().y()])
                        points.append([line.endpoints[1].pos().x(), line.endpoints[1].pos().y()])
                        continue
            if len(points) < 4:
                continue
            slope1 = self.get_slope([points[0][0], points[0][1], points[1][0], points[1][1]])
            slope2 = self.get_slope([points[2][0], points[2][1], points[3][0], points[3][1]])
            angle = self.get_angle_using_slopes(slope1, slope2)
            if p == "Right":
                self.angle_right = angle
            else:
                self.angle_left = angle

        self.text_left.setText("CCD angle left: " + str(self.angle_left) + "°")
        self.text_right.setText("CCD angle right: " + str(self.angle_right) + "°")

    def setup_scene(self, fname):
        # no images loaded
        if not self.first_im:
            return

        # two images open
        if self.first_im and self.second_im:
            self.tab.setup(None, self.detector)
            return

        # one image open
        if self.predicted_points is None:
            self.predicted_points = self.get_predicted_points(fname)

        # check on intersection
        critical = ["Femoral Shaft Centerline", "Femoral Neck Centerline"]
        for key, value in self.predicted_points.items():  # key - Left/Right
            points = list() #save start and end of line points
            for c in critical:
                for key2, value2 in value.items():  # key - Line(Name)
                    if c in key2:
                        if value2 is None:
                            continue
                        points.append([int(value2[0]), int(value2[1])])
                        points.append([int(value2[2]), int(value2[3])])
            # change coordinates if lines do not intersect
            if len(points) == 4 and not doIntersect(points[0], points[1], points[2], points[3]):
                intersec = lineLineIntersection(points[0], points[1], points[2], points[3])
                x1, x2 = value["Femoral Shaft Centerline " + str(key)][0], value["Femoral Shaft Centerline " + str(key)][2]
                i = np.argmin(np.abs([x1 - intersec[0], x2 - intersec[0]]))
                value["Femoral Shaft Centerline " + str(key)][i*2] = intersec[0]
                value["Femoral Shaft Centerline " + str(key)][i*2+1] = intersec[1]

                x1, x2 = value["Femoral Neck Centerline " + str(key)][0], \
                         value["Femoral Neck Centerline " + str(key)][2]
                i = np.argmin(np.abs([x1 - intersec[0], x2 - intersec[0]]))
                value["Femoral Neck Centerline " + str(key)][i * 2] = intersec[0]
                value["Femoral Neck Centerline " + str(key)][i * 2 + 1] = intersec[1]

        for line in self.all_lines:
            self.main_scene.removeItem(line)
        self.all_lines = list()

        for key, value in self.predicted_points.items():  # key - Left/Right
            for key2, value2 in value.items():  # key - Line(Name)
                if value2 is None:
                    continue
                if "centerline" in key2 or "Centerline" in key2:
                    new_Line = LineItem(int(value2[0]), int(value2[1]), int(value2[2]), int(value2[3]), key, key2, max(self.pic.pixmap().height(), self.pic.pixmap().width()))
                    new_Line.setParentItem(self.pic)
                    self.all_lines.append(new_Line)

        for line in self.all_lines:
            self.main_scene.addItem(line)
        self.main_view.fitInView(self.pic, Qt.KeepAspectRatio)

    def resizeEvent(self, a0) -> None:
        QMainWindow.resizeEvent(self, a0)
        if self.first_im and not self.second_im:
            self.main_view.fitInView(self.main_scene.sceneRect(), Qt.KeepAspectRatio)
        elif self.first_im and self.second_im:
            self.tab.views[0].fitInView(self.tab.pic[0], Qt.KeepAspectRatio)
            self.tab.views[1].fitInView(self.tab.pic[1], Qt.KeepAspectRatio)

    def showEvent(self, a0) -> None:
        QMainWindow.showEvent(self, a0)
        if self.first_im and not self.second_im:
            self.main_view.fitInView(self.main_scene.sceneRect(), Qt.KeepAspectRatio)

    def save_scene(self, voice=False, save_text=None):
        screen = QApplication.primaryScreen()
        if self.first_im and self.second_im:
            screenshot = screen.grabWindow(self.tab.winId())
        elif self.first_im and not self.second_im:
            screenshot = screen.grabWindow(self.main_view.winId())
        else:
            print("WARNING open image first")
            return
        items = self.all_lines.copy()
        items.append(self.pic)

        totalRect = functools.reduce(operator.or_, (i.sceneBoundingRect() for i in items))
        pix = QPixmap(int(totalRect.width()), int(totalRect.height()))
        painter = QPainter(screenshot)

        #self.main_scene.render(painter, totalRect)
        # add text to screenshot

        pen = QPen(Qt.blue)
        pen.setWidth(2)
        painter.setPen(pen)

        font = QFont()
        font.setFamily('Times')
        font.setBold(True)
        font.setPointSize(17)
        painter.setFont(font)

        if not voice:
            save_text = get_save_info()
        else:
            save_text = self.save_label.text()
        painter.drawText(20, 60, save_text)

        painter.drawText(20, 100, "Right: " + str(self.angle_right) + "°      Left: " + str(self.angle_left) + "°")

        print("Saved: ", save_text)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        screenshot.save(r"D:\Uni\demonstration\screenshots/" + str(timestr) + ".png", "PNG")
        painter = None

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def click_open(self, voice=False):
        if not voice:
            fname, x = QFileDialog.getOpenFileNames(self, "Open File", r"D:\Uni\demonstration\test_dataset/", "All Files (*)")
        else:
            list_of_files = sorted(glob.glob(r"D:\Uni\demonstration\test_dataset/*.png"), key=os.path.getctime, reverse=True)
            fname = [list_of_files[self.count % len(list_of_files)].replace("\\", "/")]
            self.count += 1


        if len(fname) == 1:
            image_qt = QImage(fname[0])

            # remove old scene
            if self.first_im and not self.second_im:
                for line in self.all_lines:
                    self.main_scene.removeItem(line)
                self.main_scene.removeItem(self.pic)
                self.all_lines = list()
                self.pic = None
                self.zoom = 0
                self.predicted_points = None
                self.angle_left = "N/A"
                self.angle_right = "N/A"

            else:
                #print(self.horizontalSpacer)
                #self.main_layout.removeItem(self.horizontalSpacer)
                self.clearLayout(self.main_layout)
                self.main_view = QGraphicsView()

                # Turn off horizontal and vertical scrollbars
                self.main_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.main_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            self.main_scene = QGraphicsScene(self)
            self.main_scene.changed.connect(lambda: self.updateAngle(self.all_lines))
            self.main_view.setScene(self.main_scene)

            # load new image
            self.pic = QGraphicsPixmapItem()
            self.pic.setPixmap(QPixmap.fromImage(image_qt))

            #self.pic.setFlag(QGraphicsItem.ItemIsMovable)
            #self.pic.setFlag(QGraphicsItem.ItemIsSelectable)
            self.first_im = True
            self.second_im = False

            # add pic to label
            self.main_scene.addItem(self.pic)
            self.setup_scene(fname[0])
            print("Scene is set up")
            self.main_layout.addWidget(self.main_view)

            self.main_view.fitInView(self.main_scene.sceneRect(), Qt.KeepAspectRatio)
            self.update()
            # next images need to create scene and view first
        elif len(fname) == 2:
            self.clearLayout(self.main_layout)
            self.tab = Tab2()
            self.main_layout.addWidget(self.tab)
            self.tab.openImage(fname, self.detector)
            # automatic angle calculation
            self.tab.scene[0].changed.connect(lambda: self.updateAngle(self.tab.all_lines[0]))
            self.tab.scene[1].changed.connect(lambda: self.updateAngle(self.tab.all_lines[1]))
            self.first_im, self.second_im = True, True
            #self.angle_right, self.angle_left = self.tab.angle_right, self.tab.angle_left

        self.text_left.setText("CCD angle left: " + str(self.angle_left) + "°")
        self.text_right.setText("CCD angle right: " + str(self.angle_right) + "°")

    def click_right(self):
        if self.zoom == 1:
            if not self.second_im:
                self.main_view.fitInView(self.pic, Qt.KeepAspectRatio)
            else:
                self.tab.zoom_out()
            self.zoom_right.setText("Zoom Right")

            self.zoom = 0
        else:
            #self.text_right.setText("CCD angle right: "+ str(self.angle_right)+ "°")
            if self.second_im:
                self.tab.maximize_right()
                self.tab.views[0].fitInView(self.tab.pic[0], Qt.KeepAspectRatio)
                self.tab.views[1].fitInView(self.tab.pic[1], Qt.KeepAspectRatio)
            else:
                max_point = [0, self.pic.pixmap().height()]
                for line in self.all_lines:
                    if "Right" == line.position:
                        max_point[0] = max(max_point[0],
                                           max(line.endpoints[0].pos().x(), line.endpoints[1].pos().x()))
                        max_point[1] = min(max_point[1],
                                           min(line.endpoints[0].pos().y(), line.endpoints[1].pos().y()))

                bounding_rect = QRectF(0, max_point[1] - 100, max_point[0] + 50, self.pic.pixmap().height() - (max_point[1] - 150))

                self.main_view.fitInView(bounding_rect, Qt.KeepAspectRatio)

            if self.zoom == -1:
                self.zoom_left.setText("Zoom Left")

            self.zoom = 1
            self.zoom_right.setText("Zoom out")

    def click_left(self):
        if self.zoom == -1:
            if not self.second_im:
                self.main_view.fitInView(self.pic, Qt.KeepAspectRatio)
            else:
                self.tab.zoom_out()
            self.zoom_left.setText("Zoom Left")

            self.zoom = 0
        else:
            if self.second_im:
                self.tab.maximize_left()
                print("trying to fit view")
                #self.update()
                #self.tab.scene[0].update()
                #self.tab.scene[1].update()
                #self.tab.views[0].updateScene(self.tab.scene[0].sceneRect())
                #self.tab.views[1].update()
                self.tab.views[0].fitInView(self.tab.pic[0], Qt.KeepAspectRatio)
                self.tab.views[1].fitInView(self.tab.pic[1], Qt.KeepAspectRatio)
            else:
                self.text_left.setText("CCD angle left: "+ str(self.angle_left) + "°")
                # find neck Centerline point on the most left, - 50 and corner x/y
                # oder find minimal x and y coordinates
                min_point = [self.pic.pixmap().width(), self.pic.pixmap().height()]
                max_point = [0, 0]
                for line in self.all_lines:
                    if "Left" == line.position:
                        min_point[0] = min(min_point[0], min(line.endpoints[0].pos().x(), line.endpoints[1].pos().x()))
                        min_point[1] = min(min_point[1], min(line.endpoints[0].pos().y(), line.endpoints[1].pos().y()))
                        max_point[0] = max(max_point[0],
                                              max(line.endpoints[0].pos().x(), line.endpoints[1].pos().x()))
                        max_point[1] = max(max_point[1],
                                              max(line.endpoints[0].pos().y(), line.endpoints[1].pos().y()))
                bounding_rect = QRectF(min_point[0] - 100, min_point[1] - 100, self.pic.pixmap().width() - (min_point[0] - 100) + 50, self.pic.pixmap().height() - (min_point[1] - 100) + 50)

                self.main_view.fitInView(bounding_rect, Qt.KeepAspectRatio)

            if self.zoom == 1:
                self.zoom_right.setText("Zoom Right")

            self.zoom = -1
            self.zoom_left.setText("Zoom out")



    def get_predicted_points(self, fname):
        pred_points = runner(self.detector, fname)

        # Angle calculation using only Shaft and Neck Centerlines
        # Right Side
        if pred_points["Right"]["Femoral Shaft Centerline Right"] is not None \
                and pred_points["Right"]["Femoral Neck Centerline Right"] is not None:
            right_shaft_slope = self.get_slope(pred_points["Right"]["Femoral Shaft Centerline Right"])
            right_neck_slope = self.get_slope(pred_points["Right"]["Femoral Neck Centerline Right"])
            angle_right = self.get_angle_using_slopes(right_shaft_slope, right_neck_slope)
            self.angle_right = angle_right

        # Left Side
        if pred_points["Left"]["Femoral Shaft Centerline Left"] is not None \
                and pred_points["Left"]["Femoral Neck Centerline Left"] is not None:
            left_shaft_slope = self.get_slope(pred_points["Left"]["Femoral Shaft Centerline Left"])
            left_neck_slope = self.get_slope(pred_points["Left"]["Femoral Neck Centerline Left"])
            angle_left = self.get_angle_using_slopes(left_shaft_slope, left_neck_slope)
            self.angle_left = angle_left
        return pred_points

    def get_slope(self, points):
        x1, y1, x2, y2 = points
        slope = (y2 - y1)/(x2 - x1 + np.finfo(float).tiny)
        return slope

    def get_angle_using_slopes(self, m1, m2):
        angle = np.abs(np.arctan((m1 - m2)/(1 + m1*m2)))
        angle = np.rad2deg(angle)
        angle = 180 - angle
        angle = np.rint(angle)
        return angle

    def get_relative_coordinate(self, pos):
        coord_x = pos.x()
        coord_y = pos.y()

        img_pix_width = self.pixmap.width()
        img_pix_heigth = self.pixmap.height()

        label_width = self.image_label.width()
        label_height = self.image_label.height()

        scale_factor_width = label_width / img_pix_width
        scale_factor_height = label_height / img_pix_heigth

        relative_width_in_img_pix = coord_x / scale_factor_width
        relative_height_in_img_pix = coord_y / scale_factor_height

        relative_coordinates_in_img_pix = QPoint(relative_width_in_img_pix, relative_height_in_img_pix)

        coord_x = (coord_x -(label_width - img_pix_width) / 2)
        coord_y = coord_y - (label_height - img_pix_heigth) / 2

        relative_coordinates_in_img_pix = QPoint(coord_x, coord_y)
        return relative_coordinates_in_img_pix

def get_save_info():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    input_diag = QInputDialog()
    input_diag.setInputMode(QInputDialog.TextInput)
    input_diag.setWindowTitle("Add text")
    input_diag.setLabelText("Additional information:")
    input_diag.resize(500, 100)
    input_diag.setStyleSheet(
        """
        QLabel{
            font-size:20px;
            font-family:Arial;
        }
        QLineEdit{
            font-size:20px;
            font-family:Arial;
        }
        QPushButton{
            font-size:20px;
            font-family:Arial;
            border-style:solid;
            border-color:black;
            border-width:1px;
        }
        """
    )

    Ok = input_diag.exec_()
    if Ok:
        patient_info = input_diag.textValue()
        save_text = dt_string + "\n" + patient_info
    else:
        save_text = dt_string

    return save_text


def get_save_info_voice():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    sv = QMessageBox()
    sv.setText("Patient information")
    sv.show()
    return sv



def record_audio(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir):
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        i = 0
        while True:
            #get and save audio to wav file
            audio = r.listen(source)
            if save_file:
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                filename = os.path.join(temp_dir, f"temp{i}.wav")
                audio_clip.export(filename, format="wav")
                audio_data = filename
            else:
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_data = torch_audio

            audio_queue.put_nowait(audio_data)
            i += 1


def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose, save_file):
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(audio_data,language='english')
        else:
            result = audio_model.transcribe(audio_data)

        if not verbose:
            predicted_text = result["text"]
            result_queue.put_nowait("You said: " + predicted_text)
        else:
            result_queue.put_nowait(result)


        if save_file:
            os.remove(audio_data)
