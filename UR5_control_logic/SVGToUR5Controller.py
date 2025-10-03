# Author: Angelica Bonilla (@monacuca on Github, @abfominaya elsewhere)
# This is a basic program that allows the user to control the UR5 to draw simple SVG drawings. 
# Only supports SVG line drawings for now. No bezier or complex curves for now! Sorry :(. 
# (This is just basic example code to help you get started.)

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

# Basic method that doesn't exist in this version of Python.
# It was literally faster just to write it myself. 
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
    print(message)
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
    print(message)
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
min_z = -0.095

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
canvas_max_x = 800
canvas_min_y = 0
canvas_max_y = 1200

# Pen orientation parameters.
rx = 0.0
ry = 3.0 # Change this one if you need to.
rz = 0.0

# ----- Basic drawing commands -----
def togglePen(X1, Y1, Z):
    X = mapV(X1,canvas_min_x, canvas_max_x, min_x, max_x)
    Y = mapV(Y1,canvas_min_y, canvas_max_y, min_y, max_y)
    if (Z == max_z):
        moveL(X, Y, min_z, rx, ry, rz, a, v, s)
        time.sleep(0.5) # Toggle movement for half a second.
        return min_z
    else: 
        moveL(X, Y, max_z, rx, ry,rz, a, v, s)
        time.sleep(0.5)
        return max_z

def MoveTo(X, Y, Z):
    Xi = mapV(X,canvas_min_x, canvas_max_x, min_x, max_x)
    Yi = mapV(Y,canvas_min_y, canvas_max_y, min_y, max_y)
    distance = dist(Xp, Yp, Xi, Yi)

    # Move to path:
    moveL(Xi,Yi,Z,rx,ry,rz, a, v, s)

    # Sleep until next command, relative to the distance of the path:
    sleep = mapV(distance, 0, max_dist, 0, 6)
    time.sleep(sleep)
    
    # Update previous points
    Xp = Xi
    Yp = Yi

def DrawLine(X1,Y1,X2,Y2):
    Z = max_z
    MoveTo(X1,Y1,Z) # Move to pen location
    togglePen(X1,Y1,Z) # Pen goes down
    MoveTo(X2,Y2,Z) # Move to second point 
    
    # Continous paths should also not have a pen raise. 
    togglePen(X2,Y2,Z) # Move pen back up to first point. 

# TODO: Extend library to support drawing other types of paths. I.e. Bezier curves
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
moveL(min_x, min_y, max_z, rx, ry, rz, a, v, s)
time.sleep(2)

# Read the SVG file
doc = minidom.parse('drawing.svg')
path_strings = [path.getAttribute('d') for path
in doc.getElementsByTagName('path')]
doc.unlink()

# Draw all lines in the SVG
for path_string in path_strings:
    path = parse_path(path_string)
    for e in path:
        if isinstance(e, Line):
            X1 = e.start.real
            Y1 = e.start.imag
            X2 = e.end.real
            Y2 = e.end.imag
            DrawLine(X1,Y1,X2,Y2)
            print("(%.2f, %.2f) - (%.2f, %.2f)" % (X1, X1, X2, X2))
# Home:
time.sleep(2)
moveL(min_x, min_y, max_z, rx, ry, rz, a, v, s)

data = s.recv(1024)

s.close()

print ("Received", repr(data))