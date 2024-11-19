Turtlebot Audio Visual Localization
==============

Turtlebot package for audio visual localization using the [Acoustic and Visual Knowledge Distillation for Contrastive Audio-Visual Localization model](https://github.com/Ehsan-Yaghoubi/Acoustic-and-Visual-Knowledge-Distillation-for-Contrastive-Audio-Visual-Localization).

Model checkpoints can be found in the original work repository and should be placed under src/checkpoints/<run_name>.

## Usage

### Terminal 1 - ROS/Robot Initialization

For laptop only execution:
```
$ roscore
```

For turtlebot execution:
```
roslaunch turtlebot_bringup minimal.launch
```

### Terminal 2 - Kinect Initialization
```
roslaunch audio_visual_localization kinect_driver_server.launch
```


### Terminal 3 - Main node
```
cd turtlebot_ws/src/audio_visual_localization/src/
rosrun audio_visual_localization neural_network_node.py --experiment_name test_run1 --model_dir checkpoints --recording_duration 3
```

### Terminal 4 - Movement control
```
rosrun audio_visual_localization move_base.py
```

If testing in isolation, use the simulated angle publisher:
```
rosrun audio_visual_localization simulated_sound_source_angle_publisher.py
```

Outputs are saved under src/checkpoints/<run_name>/viz_seed_10.

Temporary files are saved under src/data.


Check documentation for more details.