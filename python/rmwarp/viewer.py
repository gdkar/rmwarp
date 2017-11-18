
import ModernGL
from PyQt5 import Qt as Q, QtCore, QtGui, QtWidgets
from rmwarp.framer import NpFramer
from rmwarp.vocoder import Vocoder

class ViewData(Q.QObject):

    def __init__(self
        , fp
        , frame_size=2048
        , hop_size  =256
        , shaping   =2.4
        **kwargs):

        self.voc = Vocoder(fp,frame_size=frame_size,hop_size=hop_size,shaping=shaping)
