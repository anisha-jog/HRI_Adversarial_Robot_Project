# Author: Angelica Bonilla (@monacuca on Github, @abfominaya elsewhere)
# Code for Human-Robot Interaction study course at Carnegie Mellon University. 
# This probably is useless to you unless you are Lauren. Look at DrawUR5.py for actual pointers.

# ---------------------------------------------------------------------------------------
debug = True          # For Angie or Lauren. 
use_moveL = True      # For Angie or Lauren. 
training_mode = False # If training mode is on, then round turns = 2. 
# ---------------------------------------------------------------------------------------
# THIS IS THE CODE THAT WILL BE USED FOR OUR HRI STUDY. 
# STUDY PARAMS:

control = False # Is participant control or experimental. 
round_len = 60 # Robot and human take approximately 30 seconds to draw each.
round_turns = 5 # 5 turns each round. One round takes 10 minutes. 
training_turns = 1 # Will only be used if training mode is on. 
bot_line_num = 10 # Number of lines robot draws before it delegates turn to human. 

# Robot height while inactive or in home position. 
home_Z = 0.1

# If we do more than one round. 
# Ex: if we do 5 rounds, this means people will be going through 25 minutes of drawing. 
round_num = 1
break_len = 60 # Minute long breaks before next drawing starts in the round. 

# Just so you know how much it will take per person. 
total_time = round_len*2*round_turns*round_num

# NOTE: images.length() == round_num. This means 1 image (robot drawn) per round.
# Ex. If round_num == 3, we would have this:
# images = ["study/study1.svg", "study/study2.svg", "study/study3.svg"]

if (training_mode):
    images = ["study/training.svg"]
else: 
    images = ["study/test2.svg"]
# ---------------------------------------------------------------------------------------
import socket
import time
import math

# For image processing:
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom

# Note: There is an actual python interface that calculates IK and poses through RoboDK
# and an RDTE protocol for TCP/IP control of the robot. This is in case you want to directly
# write/read instructions over TCP using UR5Script. 

# Stupid method that doesn't exist in this version of Python. 
def dist(x1,y1,x2,y2):
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return d
# ----- URScript parsing helpers -----
# MoveL: Moves using unverse kinematics
def moveL(x, y, z, rx, ry, rz, a, v, s):
    # Format pose for moveL command. 
    pose = ("p[" + str(x) + ", " \
            + str(y) + ", " \
            + str(z) + ", " \
            + str(rx) + ", " \
            + str(ry) + ", " \
            + str(rz) + "]")
    
    # Format message for moveL command: 
    message = ("movel(" + pose + ", a=" + str(a) + ", v=" + str(v) + ")" + "\n")
    # Send message (i.e. pose) through socket and encode

    if (debug):
        print("Sending position to socket:" + message)
    
    s.send((message).encode('utf8'))

# MoveJ: Directly controls joint rotation of the UR5
def moveJ(x, y, z, rx, ry, rz, a, v, s):
    # Format pose for moveJ command. 
    pose = ("p[" + str(x) + ", " \
            + str(y) + ", " \
            + str(z) + ", " \
            + str(rx) + ", " \
            + str(ry) + ", " \
            + str(rz) + "]")
    
    # Format message for moveJ command: 
    message = ("movej(" + pose + ", a=" + str(a) + ", v=" + str(v) + ")"+ "\n")
    # Send message (i.e. pose) through socket and encode
    if (debug):
        print("Sending position to socket:" + message)
    s.send((message).encode('utf8'))


# Similar to map in p5.js.
# Ex: map(10, 1, 100, 1, 1000) => 100
# Ex: map(10, 2, 5, 0, 10 ) => 10
# Ex: map(10, 5, 20, 1, 2) => 1.25 
def mapV(v, min_sv, max_sv, min_dv, max_dv):
    if (v >= max_sv):
        return max_dv
    if (v <= min_sv):
        return min_dv
    res = ((v - min_sv)/(max_sv-min_sv))*(max_dv - min_dv) + min_dv 
    print(res)
    return res

# ----- Acceleration and Velocity specs for machine -----
a = 1.39626
v = 1.04719

# ----- Parameters for interpreting SVG art and mapping it to robot's coordinate system -----
# Parameters to toggle pen. 
max_z = -0.08 
min_z = -0.110

# Limits for X, Y Coordinates. 
# If these are changed it's likely robot might do protective stops, 
# one way to mitigate this is by using MoveJ as opposed to MoveL where possible. 
# Another way is to change the z value and test possible positions with the robot. 
# NOTE: see BasicUR5.py script for a quick program you can run to manually try 
# some positions for the robot. 

# Other possible values that work for me: 
# Ex: min_y = 0.45, max_y = 0.625, min_x = 0.50, max_x = 0.675
min_y = 0.3 
max_y = 0.5 
min_x = 0.3
max_x = 0.5
max_dist = dist(min_x, min_y, max_x, max_y)

# Initial points, no need to change this for now. 
Xp = min_x
Yp = min_y

# Limits for canvas size while parsing SVG Coordinates. 
# Square canvas works best unless you want to do some geometry.
canvas_min_x = 0
canvas_max_x = 210
canvas_min_y = 0
canvas_max_y = 210

# Pen orientation parameters.
rx = 0.0
ry = 3.0 # Change this one if you need to.
rz = 0.0

# ----- Basic drawing commands -----
def togglePen(X1, Y1, Z):
    X = mapV(X1,canvas_min_x, canvas_max_x, min_x, max_x)
    Y = mapV(Y1,canvas_min_y, canvas_max_y, min_y, max_y)
    if (Z == max_z):
        if (use_moveL):
            moveL(X, Y, min_z, rx, ry, rz, a, v, s)
        else:
            moveJ(X, Y, min_z, rx, ry, rz, a, v, s)
        
        time.sleep(2) # Toggle movement for half a second.
        return min_z
    else: 
        moveL(X, Y, max_z, rx, ry,rz, a, v, s)
        time.sleep(2)
        return max_z

def MoveTo(X, Y, Z):
    Xi = mapV(X,canvas_min_x, canvas_max_x, min_x, max_x)
    Yi = mapV(Y,canvas_min_y, canvas_max_y, min_y, max_y)
    # distance = dist(Xp, Yp, Xi, Yi)

    
    if (debug):
        print("Moving To: (%.2f, %.2f)" % (Xi, Yi))
    
    # Move to path:
    if (use_moveL):
        moveL(Xi,Yi,Z,rx,ry,rz, a, v, s)
    else:
        moveJ(Xi,Yi,Z,rx,ry,rz, a, v, s)

    # Sleep until next command, relative to the distance of the path:
    sleep = 2 #mapV(distance, 0, max_dist, 0, 6)
    time.sleep(sleep)
    
    # Update previous points
    Xp = Xi
    Yp = Yi

def DrawLine(X1,Y1,X2,Y2):
    Z = max_z
    MoveTo(X1,Y1,Z) # Move to pen location
    Z = togglePen(X1,Y1,Z) # Pen goes down
    MoveTo(X2,Y2,Z) # Move to second point 
    
    # Continous paths should also not have a pen raise. 
    togglePen(X2,Y2,Z) # Move pen back up to first point. 

# ----- STUDY ONLY ------
# Participants turn to draw. 
def Take_Turn(X1,Y1):
    if (control):
        # Move back home. 
        moveL(min_x, min_y, home_Z, rx, ry, rz, a, v, s)

        # Human turn to draw.
        time.sleep(round_len)
    else: 
        # Move to the next point. 
        MoveTo(X1, Y1, home_Z)
        time.sleep(round_len)


# ----- Machine Set Up ----
# Robot IP Address. 
# Note that to control the robot, it must have the corresponding DNS connection!
HOST = "169.254.154.223"

# UR5 Robot port.
PORT = 30002

# Set up socket. 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

s.send(("set_digital_out(0,True)" + "\n").encode('utf8'))
s.send(("set_digital_out(0,False)" + "\n").encode('utf8'))

# Home:
moveL(min_x, min_y, home_Z, rx, ry, rz, a, v, s)
time.sleep(2)

# Read the SVG file
for image in images:
    print("NEW DRAWING: " + image)
    doc = minidom.parse(image)
    path_strings = [path.getAttribute('d') for path
    in doc.getElementsByTagName('path')]
    doc.unlink()
    
    # Draw all lines in the SVG
    # Every 'bot_line_num' lines robot should stop and let human take its turn.
    count = 0
    for path_string in path_strings:
        path = parse_path(path_string)
        for e in path:
            if isinstance(e, Line):
                X1 = e.start.real
                Y1 = e.start.imag
                X2 = e.end.real
                Y2 = e.end.imag

                if ((count % bot_line_num) == 0): 
                    Take_Turn(X1, Y1)

                DrawLine(X1,Y1,X2,Y2)
                
                count += 1
                if (debug):
                    print("Point in SVG: (%.2f, %.2f) - (%.2f, %.2f)" % (X1, X1, X2, X2))
                    

                    
    # Move back home to start new drawing.
    if (use_moveL):
        moveL(min_x, min_y, home_Z, rx, ry, rz, a, v, s)
    else:
        moveJ(min_x, min_y, home_Z, rx, ry, rz, a, v, s)

    time.sleep(break_len)

data = s.recv(1024)

s.close()

print ("Received", repr(data))