# import the necessary packages
from threading import Thread,Lock
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
 
class PiCameraStream:
    def __init__(self, width, height) :
        self.camera = PiCamera()
        self.rawCap = PiRGBArray(self.camera)        
        self.camera.resolution = (width, height)
        self.rawCap = PiRGBArray(self.camera, size=(width, height))
        
        time.sleep(0.1)
        
        self.camera.capture(self.rawCap, format="bgr")
        self.frame = self.rawCap.array
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            self.rawCap.truncate(0)
            self.camera.capture(self.rawCap, format="bgr")
            frame = self.rawCap.array
            self.read_lock.acquire()
            self.frame = frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.camera.close()
