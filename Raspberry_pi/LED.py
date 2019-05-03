import board
import neopixel
import time
import numpy as np

class LED:
    def __init__(self,totalLED):
        self.pixels = neopixel.NeoPixel(board.D18, totalLED)
        self.led9Matrix = [
            [26,27,28,29,30,31],  # Top left corner
            [32,33,34,35,36,37,38,74,75,81],  # Top center
            [76,77,78,79,80],  # Top right corner
            [12,13,14,15,16,17,23,24,25],  # Center left
            [10,11,18,19,20,21,22,39,66,67,73],  # Center
            [64,65,68,69,70,71,72],  # Center right
            [2,3,4,5,6,7,8], # Bottom left
            [0,1,9,40,41,42,43,44,45,47,48], # Bottom center
            [46,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], # Bottom right
        ]
        
        # Positions respect to front side of board
        self.led5Matrix = [
            [10,11,18,19,20,21,22,39,66,67,73], # Center
            [32,33,34,35,36,37,38,74,75,81,26,27,28,29,30,31,76,77,78,79,80], # Top
            [64,65,68,69,70,71,72,76,77,78,79,80,46,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], # Left
            [2,3,4,5,6,7,8,0,1,9,40,41,42,43,44,45,47,48,46,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], # Bottom
            [26,27,28,29,30,31,12,13,14,15,16,17,23,24,25,2,3,4,5,6,7,8], # Right
        ]
        
        self.ledColors = [
            (8,68,56), # Green
            (65,241,214), # Green-Blue
            (4,66,133), # Blue
            ]
        
        self.previousState = -1
       

    def clear(self):
        self.pixels.fill((0,0,0))
    
    def setColors(self, leds, color):
        for led in leds:
            self.pixels[led] = color
    
    def powerOn(self, section):
        if section == 'all':
            self.previousState = -1
            
            # Green Corner
            self.setColors(self.led9Matrix[0],self.ledColors[0])
            self.setColors(self.led9Matrix[3],self.ledColors[0])
            self.setColors(self.led9Matrix[1],self.ledColors[0])    
            
            # Green-Blue Section
            self.setColors(self.led9Matrix[2],self.ledColors[1])
            self.setColors(self.led9Matrix[4],self.ledColors[1])
            self.setColors(self.led9Matrix[6],self.ledColors[1])
                
            # Blue Corner
            self.setColors(self.led9Matrix[5],self.ledColors[2])
            self.setColors(self.led9Matrix[7],self.ledColors[2])
            self.setColors(self.led9Matrix[8],self.ledColors[2])
        else:
            if self.previousState != section:
                self.clear()
            
            for ledIndex in self.led5Matrix[section]:
                self.pixels[ledIndex] = (0,0,255)
                
            self.previousState = section
        self.pixels.show()
