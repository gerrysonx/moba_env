# moba_env
gamecore is the folder containing all the moba game logic, 
gym_moba is the folder containing the interface needed to integrate the gamecore logic into gym env.

Depends:
1. go
2. gym env
3. tensorflow 1.14.0
4. golang opengl support, need to run: 
1) sudo apt-get install libgl1-mesa-dev libxinerama-dev libxcursor-dev libxi-dev xorg-dev
2) go get github.com/go-gl/gl/v4.1-core/gl 
3) go get github.com/go-gl/glfw/v3.2/glfw
4) go get github.com/go-gl/mathgl/mgl32
5) go get github.com/ungerik/go3d/vec3

Usage:
1. Enter gamecore folder, run:go build ./ 
2. Enter gym_moba, run:pip install -e gym_moba --user

3. Enter gym_moba/gym_moba/envs, edit moba_env.py, change subprocess.Popen gamecore path to your local compiled gamecore folder.
