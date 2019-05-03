#from WebcamStream import WebcamStream
#from PiCameraStream import PiCameraStream
from HeadPoseEstimator import HeadPoseEstimator
import dlib
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

if __name__ == "__main__":
    print("setting up...")
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    #stream = PiCameraStream(width=640, height=480).start()
    
    # Setup Model
    # This a pretrained model obtained from
    # https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
    landmark = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark)
    
    estimator = HeadPoseEstimator(detector, predictor)
    
    print("starting capture...")
    
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
     
        image = estimator.process(image)
     
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    cv2.destroyAllWindows()
    camera.close()

