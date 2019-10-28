# moba_env
gamecore is the folder containing all the moba game logic
gym_moba is the folder containing the interface needed to integrate the gamecore logic into gym env.

Depends:
1. go
2. gym env
3. tensorflow 1.14.0

Usage:
1. Enter gamecore folder, run go build ./ 
2. Enter gym_moba, run python ./setup.py
3. Enter gym_moba/gym_moba/envs, edit moba_env.py, change subprocess.Popen gamecore path to your local compiled gamecore folder.
