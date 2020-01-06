package core

import (
	"./nn"
	"github.com/ungerik/go3d/vec3"
)

type Lusian struct {
	Hero
	nn.Model
	action_type    uint8
	last_inference float64
	inference_gap  float64
}

func (hero *Lusian) Tick(gap_time float64) {
	game := &GameInst
	if game.ManualCtrlEnemy {
		hero.ManualCtrl(gap_time)
		return
	}

	now_seconds := game.LogicTime
	pos := hero.Position()

	isEnemyNearby, enemy := CheckEnemyNearby(hero.Camp(), hero.ViewRange(), &pos)
	if isEnemyNearby {
		pos_enemy := enemy.Position()

		canAttack := CanAttackEnemy(hero, &pos_enemy)

		if canAttack {
			NormalAttackEnemy(hero, enemy)
		}

		// Check enemy and self distance
		// dist := vec3.Distance(&pos_enemy, &pos)
		dir_a := enemy.Position()
		dir_b := hero.Position()
		var dir vec3.T
		need_move := true
		if true { // dist > enemy.AttackRange()+35
			if canAttack == false {
				// March towards enemy
				dir = vec3.Sub(&dir_a, &dir_b)
				//		LogStr(fmt.Sprintf("Lusian need move toward enemy:%v, attack range:%v, dist:%v time:%v", enemy.GetId(), enemy.AttackRange(), dist, game.LogicTime))
			} else {
				need_move = false
			}
		} else {
			// March to the opposite direction of enemy

			has_clear_dir, clear_dir := GetEnemyClearDir(hero.Camp(), &pos)
			if has_clear_dir {
				dir = clear_dir
			} else {
				dir = vec3.Sub(&dir_b, &dir_a)
			}
			//	LogStr(fmt.Sprintf("Lusian need move toward enemy:%v, attack range:%v, dist:%v, has_clear_dir:%v, time:%v",
			//		enemy.GetId(), enemy.AttackRange(), dist, has_clear_dir, game.LogicTime))

		}

		if need_move {
			dir.Normalize()
			hero.SetDirection(dir)

			dir = dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)
			// dist_after_move := vec3.Distance(&pos_enemy, &pos)
			//	LogStr(fmt.Sprintf("Lusian need move toward enemy:%v, attack range:%v, dist:%v, dist_after_move:%v, time:%v",
			//		enemy.GetId(), enemy.AttackRange(), dist, dist_after_move, game.LogicTime))
		}

		return

		// Update hero action type from nn
		if (hero.last_inference + hero.inference_gap) < float64(now_seconds) {
			game_state := game.GetGameState(true)
			hero.action_type = hero.SampleAction(game_state)
			// fmt.Printf("max val idx is:%d, input:%v, output:%v, input_val:%v\n", max_idx, game.GetGameState(), predict[0], input_val)
			hero.last_inference = now_seconds

		}

		if hero.action_type == 0 {
			// Move towards enemy.
			// March towards target direction
			dir_a := enemy.Position()
			dir_b := hero.Position()
			dir := vec3.Sub(&dir_a, &dir_b)
			dir.Normalize()
			hero.SetDirection(dir)

			dir = dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)

		} else {
			// Set new direction accordingly
			switch hero.action_type {
			case 1:
				hero.direction[0] = -1.0
				hero.direction[1] = -1.0
			case 2:
				hero.direction[0] = 0.0
				hero.direction[1] = -1.0
			case 3:
				hero.direction[0] = 1.0
				hero.direction[1] = -1.0
			case 4:
				hero.direction[0] = -1.0
				hero.direction[1] = 0.0
			case 5:
				hero.direction[0] = 1.0
				hero.direction[1] = 0.0
			case 6:
				hero.direction[0] = -1.0
				hero.direction[1] = 1.0
			case 7:
				hero.direction[0] = 0.0
				hero.direction[1] = 1.0
			case 8:
				hero.direction[0] = 1.0
				hero.direction[1] = 1.0
			}

			hero.direction.Normalize()
			// March towards target direction
			unscaled_dir := hero.Direction()
			dir := unscaled_dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)
		}

	} else {
		//	panic("Not supposed to be here")
	}
}
