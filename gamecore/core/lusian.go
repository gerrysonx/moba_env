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

var lusian_template Lusian

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
			// Check if time to make hurt
			if (hero.LastAttackTime() + hero.AttackFreq()) < float64(now_seconds) {
				// Make damage
				dir_a := enemy.Position()
				dir_b := hero.Position()
				dir := vec3.Sub(&dir_a, &dir_b)
				bullet := new(Bullet).Init(hero.Camp(), hero.Position(), dir, hero.Damage())
				game.AddUnits = append(game.AddUnits, bullet)

				hero.SetLastAttackTime(now_seconds)
			}
		}

		// Check enemy and self distance
		dist := vec3.Distance(&pos_enemy, &pos)
		dir_a := enemy.Position()
		dir_b := hero.Position()
		var dir vec3.T
		need_move := true
		if dist > enemy.AttackRange() {
			if canAttack == false {
				// March towards enemy
				dir = vec3.Sub(&dir_a, &dir_b)
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

		}

		if need_move {
			dir.Normalize()
			hero.SetDirection(dir)

			dir = dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)
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

func (hero *Lusian) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)
	if lusian_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		lusian_template.InitFromJson("./cfg/lusian.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = lusian_template
	pos_x := a[1].(float32)
	pos_y := a[2].(float32)

	InitHeroWithCamp(hero, wanted_camp, pos_x, pos_y)
	hero.action_type = 0
	hero.inference_gap = 0.1
	hero.SetLastAttackTime(0.0)
	hero.PrepareInput()
	return hero
}

func (hero *Lusian) UseSkill(skill_idx uint8, a ...interface{}) {

	switch skill_idx {
	case 0:
		// Zap!

	case 1:
		// Flame Chompers!

	case 2:
		// Super Mega Death Rocket!

	}

}
