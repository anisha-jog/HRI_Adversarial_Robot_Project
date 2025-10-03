# LetsDrawTogetherUR5: 
Basic scripts written for Let's Draw Together study as part of Angelica Bonilla's and Lauren Kung's final project for Human-Robot Interaction. 

This repository includes basic scripts for remote/TCP control of UR robots and the scripts and SVG files specifically used for our study. 

This also serves as some sample code to help people who might want to control the UR5 robot from their computer using a very simple TCP server. 

It's important to know a couple of things:
- This repo manually sends UrScript commands over TCP. Universal Robots already has a fantastic package for remote Python work known as RDTE, for more complicated real-time robot control, you might want to check that out. 

- If you want to integrate websockets, you can create a JS/Node client and send messages using Python servers similar to this. (Note: I will add examples eventually.)

- You can also use RoboDK, which supports various UR robots. 

If you are interested in the script we used for our study, see: HRIStudy.Py

This repo was programmed and managed by:

Angelica Bonilla Fominaya 

Carnegie Mellon University. 2022
