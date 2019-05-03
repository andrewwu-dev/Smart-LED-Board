# **LED Board**
LED pledge board that lights up based on where you look. The LED's and picamera are controlled by a raspberry pi.

## **Installation**
In order to run the script for the LED board. Download [vnc](https://www.realvnc.com/en/connect/download/viewer/)
and connect to raspberry pi ip 192.168.45.138.

Username: raspberry

Password: pi

Don't be alaramed by the resolution.

Open terminal and execute `sudo python3 Main.py`. I've dragged all the files of the code into the pi folder.

The program will take some time to set up the detector and camera. *Need sudo because LED control requires.
root privileges*

## **Libraries Used**
-   [dlib](http://dlib.net/) - Face detection and facial feature prediction

-   opencv - image processing (i.e. convert grayscale to reduce noise, resize images to reduce searching area).
    head pose estimation.
    
-   [picamera](https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/) - 
    capture video
    
## **Notes**
-   Camera is capturing at 5 fps due to the fact that the facial detection algorithim is very computationally heavy 
    (raspberry pi slows down).

-   Program calculates pitch, yaw, row of head and check if they are within preset ranges to determine
    which section you are looking at.

-   There are five sections that can be controlled by head (left,right,up,down,center).

-   when the entire board lights up we control 9 sections (add in the 4 corners).

-   Camera dimension and distortion matrix are naively approximated. [refer to this site](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)

## **Todo**
-   Wires are currently duct taped together. Need something more secure.

-   Make the LED's display in a gradient color (may need to convert LED.py into a thread).

-   Add a loading animation if program can't detect your head (preferably purple colored).

-   Make the script automatically run when the rapsberry by starts up.

-   Make a marking on the floor to indicate where to stand.