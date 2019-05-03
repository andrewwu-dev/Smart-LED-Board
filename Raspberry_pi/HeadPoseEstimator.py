# Imports:
import math
import numpy as np
# Facial feature detection
import dlib
# Image processing
import cv2
from imutils import face_utils
from LED import LED
import time

class HeadPoseEstimator:
    def __init__(self,detector, predictor):
        
        self.detector = detector
        self.predictor = predictor
        self.ledController = LED(totalLED=82)
        self.counter = time.time()

    # Calculate the pitch, yaw, roll of head
    # @Params: a video frame, dictionary of detected facical feature points
    def find_face_orientation(self,frame, landmarks):
        # (height, width, color_channel)
        size = frame.shape

        # 2D Image points of detected features based on 68 point landmark detection
        # https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
        image_points = np.array([
            (landmarks[30][0], landmarks[30][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),  # Chin
            (landmarks[45][0], landmarks[45][1]),  # Left eye left corner
            (landmarks[36][0], landmarks[36][1]),  # Right eye right corner
            (landmarks[54][0], landmarks[54][1]),  # Left Mouth corner
            (landmarks[48][0], landmarks[48][1])  # Right mouth corner
        ], dtype="double")

        # Generic 3D world coordinates
        # These coordinates were predetermined using a 3D face modeling software
        # Universally accepted for head post estimation
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Intrinsic parameters of camera
        center = (size[1] / 2, size[0] / 2)
        # Approximate the focal length we do not calibrate the camera beforehand
        # Assuming that radial/lens distortion does not exist
        # width / depth factor
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # Make a 4x1 matrix of 0.
        # If dist_coeff is null (set to all 0) solvePnP will assume no lens distortion.
        dist_coeffs = np.zeros((4, 1))
        # Estimate head pose orientation
        (_ , rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # Define a 3D axis
        axis = np.float32([[500, 0, 0],
                           [0, 500, 0],
                           [0, 0, 500]])

        # Matrices needed to compute angles
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))

        # Calculate pitch yaw and roll
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        # Extract info
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        # Convert to degrees
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return (int(roll), int(pitch), int(yaw))

    def light_board(self,rotation):
        roll, pitch, yaw = rotation
        
        pitch = pitch * -1
        
        threshL = -5
        threshR = 29
        # Bottomssssss
        threshUp = -2
        # Top
        threshDown = -8

        # Centered Head
        if yaw >= threshL and yaw <= threshR and pitch >= threshDown and pitch <= threshUp:
            self.ledController.powerOn(0)
        else:
            # Horizontal
            if pitch >= threshDown and pitch <= threshUp:
                if yaw < 0:
                    self.ledController.powerOn(2)
                elif yaw > 0:
                    self.ledController.powerOn(4)
            # Vertical
            elif yaw >= threshL and yaw <= threshR:
                if pitch > 0:
                    self.ledController.powerOn(1)
                elif pitch < 0:
                    self.ledController.powerOn(3)
                    
    def process(self, frame):
        # Convert screen capture to black-white image to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray, 0)
        
        if len(faces) != 1:
            if time.time() - self.counter < 5:
                time.sleep(1)
            else:
                #Turn on all leds
                self.ledController.powerOn('all')
                # Reset counter after five seconds
                self.counter = time.time()
            
            return frame
        else:
            # Instantly reset counter if face is detected again
            self.counter = time.time()
            
        #for face in faces:
        face = faces[0]
        # Get face landmarks i.e. right_eye, nose, mouth, etc
        facial_landmarks = self.predictor(gray, face)
        facial_landmarks = face_utils.shape_to_np(facial_landmarks)

        # Attempt to estimate head pose
        rotation = self.find_face_orientation(frame, facial_landmarks)

        self.light_board(rotation)
        
        return frame
            
        

