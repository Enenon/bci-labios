usar_modelo = True
if usar_modelo:
    try:
        from keras.models import load_model
        from keras.optimizers import Adam
    except:
        usar_modelo = False
        print('Erro ao importar Keras.')
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPolygon
# from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import QPolygonF
from PyQt5.QtCore import *
import sys
import os
import numpy as np
from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
from pylsl import StreamInlet, resolve_stream, resolve_byprop
from time import sleep, time
import matplotlib.pyplot as plt
from random import choice
import threading
import zmq
#from random import uniform
import random
import socket
import datetime
import pandas as pd

PORTA_UNITY = 5555
PORTA_UDP_UNITY = 12346

def aplicar_estilo_escuro(janela):
        qss = """
        QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI', Arial; }
        QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; background-color: #2b2b2b; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; background-color: #2b2b2b; color: #aaaaaa; }
        QPushButton { background-color: #3c3f41; border: 1px solid #555; border-radius: 4px; padding: 5px; color: white; }
        QPushButton:hover { background-color: #484b4d; }
        QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }
        QTabBar::tab { background: #2b2b2b; color: #888888; padding: 8px 25px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: bold; }
        QTabBar::tab:selected { background: #3c3f41; color: #ffffff; border-bottom: 3px solid #00bcd4; }
        QComboBox, QSpinBox, QDoubleSpinBox { background: #3c3f41; border: 1px solid #555; padding: 3px; color: white; }
        QProgressBar { border: 1px solid #555; text-align: center; color: white; }
        QProgressBar::chunk { background-color: #00bcd4; }
        QCheckBox { color: white; spacing: 5px; }
        QCheckBox::indicator { width: 15px; height: 15px; }
        """
        janela.setStyleSheet(qss)