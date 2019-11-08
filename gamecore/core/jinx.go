package core

import (
	"./nn"
	"github.com/ungerik/go3d/vec3"
)

type Jinx struct {
	Hero
	nn.Model
	action_type    uint8
	last_inference float64
	inference_gap  float64
}

var jinx_template Jinx

func (hero *Jinx) Tick(gap_time float64) {
	game := &GameInst
	now_seconds := game.LogicTime
	pos := hero.Position()

	isEnemyNearby, enemy := CheckEnemyNearby(hero.Camp(), hero.ViewRange(), &pos)
	if isEnemyNearby {
		pos_enemy := enemy.Position()

		// Update hero action type from nn
		if (hero.last_inference + hero.inference_gap) < float64(now_seconds) {
			game_state := game.GetGameState(true)
			hero.action_type = hero.SampleAction(game_state)
			// fmt.Printf("max val idx is:%d, input:%v, output:%v, input_val:%v\n", max_idx, game.GetGameState(), predict[0], input_val)
			hero.last_inference = now_seconds

		}

		if hero.action_type == 0 {
			isEnemyCanAttack := CanAttackEnemy(hero, &pos_enemy)

			if isEnemyCanAttack {
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
			dir := hero.Direction()
			dir = dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)
		}

	} else {
		//	panic("Not supposed to be here")
	}
}

func (hero *Jinx) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)
	if jinx_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		jinx_template.InitFromJson("./cfg/jinx.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = jinx_template
	pos_x := a[1].(float32)
	pos_y := a[2].(float32)

	InitHeroWithCamp(hero, wanted_camp, pos_x, pos_y)
	hero.action_type = 0
	hero.inference_gap = 0.1
	hero.lastAttackTime = 0.0
	hero.PrepareInput()
	return hero
}

func (hero *Jinx) UseSkill(skill_idx uint8, a ...interface{}) {

	switch skill_idx {
	case 0:
		// Zap!

	case 1:
		// Flame Chompers!

	case 2:
		// Super Mega Death Rocket!

	}

}
