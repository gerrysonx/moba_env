package main

import (
	"flag"
	"fmt"
	_ "image/png"
	"os"
	"time"

	"./core"
	"./render"
)

func main() {
	_target_frame_gap_time := flag.Float64("frame_gap", 0.03, "")
	_fix_update := flag.Bool("fix_update", true, "a bool")
	_run_render := flag.Bool("render", true, "a bool")
	_input_gap_time := flag.Float64("input_gap", 0.1, "")
	_manual_enemy := flag.Bool("manual_enemy", false, "a bool")
	_gym_mode := flag.Bool("gym_mode", false, "a bool")
	_debug_log := flag.Bool("debug_log", false, "a bool")
	file_handle, _err := os.Create("mobacore.log")
	if _err != nil {
		fmt.Println("Create log file failed.")
		return
	}

	defer file_handle.Close()
	if *_debug_log {
		core.LogHandle = file_handle
	} else {
		core.LogHandle = nil
	}
	flag.Parse()
	core.LogStr(fmt.Sprintf("main is called"))

	core.GameInst = core.Game{}
	core.GameInst.Init()
	core.GameInst.ManualCtrlEnemy = *_manual_enemy
	if *_run_render {
		render.RendererInst = render.Renderer{}
		render.RendererInst.InitRenderEnv(&core.GameInst)
		defer render.RendererInst.Release()
	}

	now := time.Now()
	_before_tick_time := float64(now.UnixNano()) / 1e9
	_after_tick_time := _before_tick_time
	_logic_cost_time := _before_tick_time
	_gap_time := float64(0)

	_action_stamp := int(0)
	// Set target frame time gap

	if *_fix_update {
		_last_input_time := core.GameInst.LogicTime
		var action_code int   // raw action code
		var action_code_0 int // action code
		var action_code_1 int // move dir code
		var action_code_2 int // skill dir code
		for {
			// Process input from user
			if core.GameInst.LogicTime > _last_input_time+*_input_gap_time && *_gym_mode {
				_last_input_time = core.GameInst.LogicTime

				// Output game state to stdout
				game_state_str := core.GameInst.DumpGameState()
				// core.LogBytes(file_handle, game_state_str)
				fmt.Printf("%d@%s\n", _action_stamp, game_state_str)
				fmt.Scanf("%d\n", &action_code)
				//action_code = 4352
				_action_stamp = action_code >> 16

				action_code_0 = (action_code >> 12) & 0xf
				action_code_1 = (action_code >> 8) & 0xf
				action_code_2 = (action_code >> 4) & 0xf

				core.GameInst.HandleMultiAction(action_code_0, action_code_1, action_code_2)
				if 9 == action_code_0 {
					// Instantly output
					_last_input_time = 0
				}
			}

			if false == *_gym_mode {
				game_state_str := core.GameInst.DumpGameState()
				// core.LogBytes(file_handle, game_state_str)
				fmt.Printf("%d@%s\n", _action_stamp, game_state_str)
				gap_time_in_nanoseconds := *_target_frame_gap_time * float64(time.Second)
				time.Sleep(time.Duration(gap_time_in_nanoseconds))
			}
			core.GameInst.Tick(*_target_frame_gap_time)

			// Draw logic units on output image
			if *_run_render {
				render.RendererInst.Render()
			}

		}
	} else {
		for {
			_gap_time = _after_tick_time - _before_tick_time

			// fmt.Printf("------------------------------\nMain game loop, _before_tick_time is:%v\n", _before_tick_time)
			_before_tick_time = _after_tick_time
			// Do the tick
			core.GameInst.Tick(_gap_time)
			// Draw logic units on output image
			if *_run_render {
				render.RendererInst.Render()
			}

			now = time.Now()
			_logic_cost_time = (float64(now.UnixNano()) / 1e9) - _before_tick_time

			if _logic_cost_time > *_target_frame_gap_time {
			} else {
				gap_time_in_nanoseconds := (*_target_frame_gap_time - _logic_cost_time) * float64(time.Second)
				time.Sleep(time.Duration(gap_time_in_nanoseconds))
			}
			now = time.Now()
			_after_tick_time = float64(now.UnixNano()) / 1e9
			// fmt.Printf("Main game loop, _after_tick_time is:%v, _gap_time is:%v, logic cost time is:%v\n------------------------------\n", _after_tick_time, _gap_time, _logic_cost_time)
		}
	}

}
