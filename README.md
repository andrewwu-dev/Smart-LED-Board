# **LED Board**
LED pledge board that lights up based on where you look. The LED's and picamera are controlled by a raspberry pi.

[![](http://img.youtube.com/vi/4Uyygn34Lp8/0.jpg)](http://www.youtube.com/watch?v=4Uyygn34Lp8 "Demo 1")

[![](http://img.youtube.com/vi/ELXCX4X0leI/0.jpg)](http://www.youtube.com/watch?v=ELXCX4X0leI "Demo 2")

## **Libraries Used**
-   [dlib](http://dlib.net/) - Face detection and facial feature prediction

-   opencv - image processing (i.e. convert grayscale to reduce noise, resize images to reduce searching area).
    head pose estimation.
    
-   [picamera](https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/) - 
    capture video
    
## **Notes**

-   Program calculates pitch, yaw, row of head and check if they are within preset ranges to determine
    which section you are looking at.

-   There are five sections that can be controlled by head (left,right,up,down,center).

-   when the entire board lights up we control 9 sections (add in the 4 corners).

-   Camera dimension and distortion matrix are approximated. [refer to this site](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)

## **Todo**

-   Make the LED's display in a gradient color (may need to convert LED.py into a thread).

-   Add a loading animation if program can't detect your head (preferably purple colored).

-   Make the script automatically run when the rapsberry by starts up.
