usar_modelo = True
if usar_modelo:
    from keras.models import load_model
    from tensorflow.keras.optimizers import Adam
from PyQt5 import QtCore, QtGui, QtWidgets

# from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtWidgets import * 
import sys
import numpy as np
from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
from pylsl import StreamInlet, resolve_stream, resolve_byprop
from time import sleep
import matplotlib.pyplot as plt
from random import choice
import threading
import zmq

