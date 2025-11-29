# HRI_Adversarial_Robot_Project
```
Project_Root/
├── README.md
├── tests/
│   ├── index.md
│   └── setup.md
├── ros2_ws/
│   ├── main.py
│   ├── components/
│   │   └── button.js
│   └── utils/
│       └── helpers.js
└── tests/
    └── test_main.py
```


## Running ROS

Start the software with 
`ros2 launch ur5_draw ur5_draw.launch.py`

**Optional** Set open the rviz config at `ros2_ws/src/ur5_draw/ur5_draw/config/draw.rviz`

Click the play button on the UR pendant

### Drawing Images

Can be done two different ways
1. Call the `'draw_strokes'` action server either by
    - Calling `ros2 run ur5_draw test IMG_PATH`
    - importing `DrawActionClient` from `test_action_client.py`
2. Import `Draw` from `draw_node.py`