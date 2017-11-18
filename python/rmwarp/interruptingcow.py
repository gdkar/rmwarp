
import PyQt5.Qt as Q, PyQt5.QtCore, PyQt5.QtGui, PyQt5.QtWidgets, PyQt5.QtNetwork
import signal, sys, socket

class SignalWakeupHandler(Q.QAbstractSocket):

    signalReceived = Q.pyqtSignal(int)
    def __init__(self, parent=None):
        super(SignalWakeupHandler,self).__init__(Q.QAbstractSocket.UdpSocket, parent)
        self.old_fd = None
        # Create a socket pair
        self.wsock, self.rsock = socket.socketpair(type=socket.SOCK_DGRAM)
        # And let Python write on the other end
        self.wsock.setblocking(False)
        # Let Qt listen on the one end
        self.setSocketDescriptor(self.rsock.fileno())
        # First Python code executed gets any exception from
        # the signal handler, so add a dummy handler first
        self.readyRead.connect(lambda : None)
        # Second handler does the real handling
        self.readyRead.connect(lambda:self.readData(1))#self._readSignal)
    def __enter__(self):
        self.old_fd = signal.set_wakeup_fd(self.wsock.fileno())
        return self

    def __exit__(self, *exc):
        old_fd, self.old_fd = self.old_fd, None
        if old_fd is not None:
            signal.set_wakeup_fd(old_fd)
        return False

#    def __del__(self):
#        self.close()

##    def close(self):
        # Restore any old handler on deletion

