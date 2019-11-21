## Introduction
1. moba_env is a mini moba game, which can be integrated into gym env.
1. You can use this env to train AI for moba game.
1. All moba logic is implemented in go lang, which is easy to maintain.

## Structure
1. gamecore is the folder containing all the moba game logic, 
1. gym_moba is the folder containing the interface needed to integrate the gamecore logic into gym env.

## Depends:
1. go
1. gym env
1. tensorflow 1.14.0
1. opencv2, you can install by pip:  
`pip install opencv-python --user`  
1. golang opengl support, need to run the following commands:   
`sudo apt-get install libgl1-mesa-dev libxinerama-dev libxcursor-dev libxi-dev xorg-dev`  
`go get github.com/go-gl/gl/v4.1-core/gl`  
`go get github.com/go-gl/glfw/v3.2/glfw`  
`go get github.com/go-gl/mathgl/mgl32`  
`go get github.com/ungerik/go3d/vec3`  
6. golang tensorflow support, follow the following steps:  
download TensorFlow C library, and extract to local lib and include path, e.g. `sudo tar -C /usr/local -xzf (downloaded file)`
reference:https://www.tensorflow.org/install/lang_c  
execute command in terminal:`go get github.com/tensorflow/tensorflow/tensorflow/go`  

## Usage:
1. Enter gamecore folder, run:`go build ./`  
1. Under moba_env folder, run:`pip install -e gym_moba --user`  
1. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib.  
1. Under moba_env folder, run python3 ../ppo.py, if you want to play the game, just change the is_train = False in main function.  
1. To modify test skill, please make modification as follows:   
in vi.go file, 'UseSkill' function, enable your tuning skill, disable other skills by short return   
in ppo_multihead.py file, 'predict' function, disable the direction mask of your tuning skill by commenting the 'actions[2] = -1' line, while maintaining masks of other skills.   

