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
1. opencv2, you can install by pip:pip install opencv-python --user
1. golang opengl support, need to run the following commands: 
5.1 `sudo apt-get install libgl1-mesa-dev libxinerama-dev libxcursor-dev libxi-dev xorg-dev`  
5.2 `go get github.com/go-gl/gl/v4.1-core/gl`  
5.3 `go get github.com/go-gl/glfw/v3.2/glfw`  
5.4 `go get github.com/go-gl/mathgl/mgl32`  
5.5 `go get github.com/ungerik/go3d/vec3`  
6. golang tensorflow support, follow the following steps:
6.1 download TensorFlow C library, and extract to local lib and include path, e.g. sudo tar -C /usr/local -xzf (downloaded file)
reference:https://www.tensorflow.org/install/lang_c
6.2 execute command in terminal:`go get github.com/tensorflow/tensorflow/tensorflow/go`

## Usage:
1. Enter gamecore folder, run:go build ./ 
1. Under moba_env folder, run:pip install -e gym_moba --user

1. Enter gym_moba/gym_moba/envs, edit moba_env.py, change subprocess.Popen gamecore path to your local compiled gamecore folder.

1. Under moba_env folder, run python3 ../ppo.py, if you want to play the game, just change the is_train = False in main function.
