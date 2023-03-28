import sys

from PyQt5.QtWidgets import QApplication

from Interface.GUI import UI


def gui_main():
    up = QApplication(sys.argv)
    window = UI(femur_detector_model_path="model_v2.0.0\checkpoints\epoch=29-step=1170.ckpt")
    up.exec()

if __name__=="__main__":
    gui_main()
