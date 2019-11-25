package core

import (
	"fmt"
	"time"

	"github.com/ungerik/go3d/vec3"
)

type Vi struct {
	Hero
}

func (hero *Vi) Chase(pos_enemy vec3.T, gap_time float64) {
	pos := hero.Position()

	// Check milestone distance
	targetPos := pos_enemy
	dist := vec3.Distance(&pos, &targetPos)
	if dist < hero.AttackRange() {
		// Already the last milestone
		hero.direction[0] = 0
		hero.direction[1] = 0
	} else {
		// March towards enemy pos
		dir := hero.Direction()
		dir = dir.Scaled(float32(gap_time))
		dir = dir.Scaled(float32(hero.speed))
		newPos := vec3.Add(&pos, &dir)
		hero.SetPosition(newPos)

		// Calculate new direction
		hero.direction = targetPos
		hero.direction.Sub(&newPos)
		hero.direction.Normalize()
	}

}

func (hero *Vi) Escape(pos_enemy vec3.T, gap_time float64) {
	pos := hero.Position()

	// Escape from enemy pos
	// Check milestone distance
	targetPos := pos_enemy
	dist := vec3.Distance(&pos, &targetPos)
	if dist > 100 {
		// Already the last milestone
		hero.direction[0] = 0
		hero.direction[1] = 0
	} else {
		dir := hero.Direction()
		dir = dir.Scaled(float32(gap_time))
		dir = dir.Scaled(float32(hero.speed))
		newPos := vec3.Add(&pos, &dir)
		hero.SetPosition(newPos)

		// Calculate new direction
		hero.direction = targetPos
		hero.direction.Sub(&newPos)
		// Revert the target direction, so we let the  hero escape
		hero.direction[0] = -hero.direction[0]
		hero.direction[1] = -hero.direction[1]
		hero.direction.Normalize()
	}
}

func (hero *Vi) Tick(gap_time float64) {
	//	fmt.Printf("Ezreal is ticking, %v gap_time is:%v\n", now, gap_time)
	game := &GameInst

	if game.ManualCtrlEnemy {
		hero.ManualCtrl(gap_time)
		return
	}

	pos := hero.Position()
	// Check milestone distance
	targetPos := hero.TargetPos()
	dist := vec3.Distance(&pos, &targetPos)
	pos_ready := false
	if dist < 4 {
		// Already the last milestone
		pos_ready = true
	}

	isEnemyNearby, enemy := CheckEnemyNearby(hero.Camp(), hero.AttackRange(), &pos)
	if isEnemyNearby && pos_ready {
		// Check if time to make hurt
		now_seconds := game.LogicTime
		if (hero.LastAttackTime() + hero.AttackFreq()) < float64(now_seconds) {
			// Make damage
			dir_a := enemy.Position()
			dir_b := hero.Position()
			dir := vec3.Sub(&dir_a, &dir_b)
			bullet := new(Bullet).Init(hero.Camp(), hero.Position(), dir, hero.Damage())
			game.AddUnits = append(game.AddUnits, bullet)

			hero.SetLastAttackTime(now_seconds)
		}
	} else {

		if pos_ready {
			// Do nothing

		} else {
			// March towards target direction
			dir := hero.Direction()
			dir = dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)

			// Calculate new direction
			hero.direction = targetPos
			hero.direction.Sub(&newPos)
			hero.direction.Normalize()
		}
	}
}

var Vi_template Vi

func (hero *Vi) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)

	if Vi_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		Vi_template.InitFromJson("./cfg/vi.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = Vi_template

	pos_x := a[1].(float32)
	pos_y := a[2].(float32)
	InitHeroWithCamp(hero, wanted_camp, pos_x, pos_y)
	hero.skillusefrequency[0] = 2
	hero.skillusefrequency[1] = 2
	hero.skillusefrequency[2] = 2
	hero.skillusefrequency[3] = 2
	return hero
}

func (hero *Vi) UseSkill(skill_idx uint8, a ...interface{}) {
	game := &GameInst
	// Check CD
	now_seconds := game.LogicTime
	if (hero.LastSkillUseTime(skill_idx) + hero.skillusefrequency[skill_idx]) > now_seconds {
		return
	}

	hero.SetLastSkillUseTime(skill_idx, now_seconds)

	switch skill_idx {
	case 0:
		return
		// Clear skilltarget pos
		// Check if has more parameters
		callback := func(skill_target *SkillTarget) {
			var unit BaseFunc
			var dir vec3.T
			unit = skill_target.hero
			dir = skill_target.dir
			ChainDamage(dir, unit.Position(), unit.Camp(), 50, 500.0)
			LogStr(fmt.Sprintf("Dir skill callback is called, dir is:%v, %v", dir[0], dir[1]))
		}

		has_more_params := len(a) > 0

		LogStr(fmt.Sprintf("UseSkill is called, has_more_params:%v, now_seconds:%v", has_more_params, now_seconds))

		skill_target := SkillTarget{}
		skill_target.callback = callback
		skill_target.trigger_time = 0

		if has_more_params {
			pos_x := a[0].(float32)
			pos_y := a[1].(float32)

			skill_target.hero = hero
			skill_target.dir[0] = pos_x
			skill_target.dir[1] = pos_y
			skill_target.dir.Normalize()
			game.AddTarget(skill_target)
			LogStr(fmt.Sprintf("AddTarget dir skill, dir is:%v, %v", pos_x, pos_y))
		} else {
			// UI mode, we're waiting for mouse to be pressed.
			hero.SetSkillTargetPos(0, 0)
			go func(hero *Vi) {
				for {
					time.Sleep(time.Duration(0.5 * float64(time.Second)))
					// Wait for left button click to select position
					skill_target_pos := hero.SkillTargetPos()
					if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
						// Use skill
						skill_target.dir[0] = skill_target_pos[0] - hero.Position()[0]
						skill_target.dir[1] = skill_target_pos[1] - hero.Position()[1]
						skill_target.dir.Normalize()
						skill_target.hero = hero
						callback(&skill_target)
						break
					}
				}
			}(hero)
		}

	case 1:

		// Overdrive
		// Slow direction
		// Clear skilltarget pos
		// Check if has more parameters
		skill_dist := float32(100.0)
		callback := func(skill_target *SkillTarget) {
			var unit BaseFunc
			unit = skill_target.hero
			AddSpeedBuff([]BaseFunc{unit}, BuffSpeedSlow)
			LogStr(fmt.Sprintf("SlowDirection is called, AddSpeedBuff calling finished."))
		}

		has_more_params := len(a) > 0

		LogStr(fmt.Sprintf("UseSkill 1 is called, has_more_params:%v, now_seconds:%v", has_more_params, now_seconds))

		skill_target := SkillTarget{}
		skill_target.callback = callback
		skill_target.trigger_time = 0

		if has_more_params {
			pos_x := a[0].(float32)
			pos_y := a[1].(float32)
			skill_target.dir[0] = pos_x
			skill_target.dir[1] = pos_y
			skill_target.dir.Normalize()
			// Find the target hero along the direction
			my_pos := hero.Position()
			_find, _enemy := CheckEnemyOnDirWithinDist(hero.Camp(), &my_pos, &skill_target.dir, skill_dist)

			if _find {
				slow_buff := _enemy.GetBuff(BuffSpeedSlow)
				if nil == slow_buff {
					skill_target.hero = _enemy
					game.AddTarget(skill_target)
					LogStr(fmt.Sprintf("UseSkill 1, AddTarget dir skill, dir is:%v, %v", pos_x, pos_y))
				}
			}
		} else {
			// UI mode, we're waiting for mouse to be pressed.
			hero.SetSkillTargetPos(0, 0)
			go func(hero *Vi) {
				for {
					time.Sleep(time.Duration(0.5 * float64(time.Second)))
					// Wait for left button click to select position
					skill_target_pos := hero.SkillTargetPos()
					if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
						// Use skill
						skill_target.dir[0] = skill_target_pos[0] - hero.Position()[0]
						skill_target.dir[1] = skill_target_pos[1] - hero.Position()[1]
						skill_target.dir.Normalize()

						my_pos := hero.Position()
						_find, _enemy := CheckEnemyOnDirWithinDist(hero.Camp(), &my_pos, &skill_target.dir, skill_dist)

						if _find {
							skill_target.hero = _enemy

							callback(&skill_target)
						}
						break
					}
				}
			}(hero)
		}

	case 2:
		return
		// Power Fist
		// Self Offset directione

		// Clear skilltarget pos
		// Check if has more parameters
		callback := func(skill_target *SkillTarget) {
			var unit BaseFunc
			var dir vec3.T
			unit = skill_target.hero
			dir = skill_target.dir
			AlterUnitPosition(dir, unit, 60)
		}

		has_more_params := len(a) > 0

		LogStr(fmt.Sprintf("UseSkill 2 is called, has_more_params:%v, now_seconds:%v", has_more_params, now_seconds))

		skill_target := SkillTarget{}
		skill_target.callback = callback
		skill_target.trigger_time = 0

		if has_more_params {
			pos_x := a[0].(float32)
			pos_y := a[1].(float32)

			skill_target.hero = hero
			skill_target.dir[0] = pos_x
			skill_target.dir[1] = pos_y
			skill_target.dir.Normalize()
			game.AddTarget(skill_target)
			LogStr(fmt.Sprintf("UseSkill 2, AddTarget dir skill, dir is:%v, %v", pos_x, pos_y))
		} else {
			// UI mode, we're waiting for mouse to be pressed.
			hero.SetSkillTargetPos(0, 0)
			go func(hero *Vi) {
				for {
					time.Sleep(time.Duration(0.5 * float64(time.Second)))
					// Wait for left button click to select position
					skill_target_pos := hero.SkillTargetPos()
					if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
						// Use skill
						skill_target.dir[0] = skill_target_pos[0] - hero.Position()[0]
						skill_target.dir[1] = skill_target_pos[1] - hero.Position()[1]
						skill_target.dir.Normalize()
						skill_target.hero = hero
						callback(&skill_target)
						break
					}
				}
			}(hero)
		}
	case 3:
		return
		// Static Field
		// Target offset

		// Clear skilltarget pos
		// Check if has more parameters
		callback := func(skill_target *SkillTarget) {
			var unit BaseFunc
			var dir vec3.T
			unit = skill_target.hero
			dir = skill_target.dir
			AlterUnitPosition(dir, unit, 100)
		}

		has_more_params := len(a) > 0

		LogStr(fmt.Sprintf("UseSkill 3 is called, has_more_params:%v, now_seconds:%v", has_more_params, now_seconds))

		skill_target := SkillTarget{}
		skill_target.callback = callback
		skill_target.trigger_time = 0

		if has_more_params {
			pos_x := a[0].(float32)
			pos_y := a[1].(float32)

			skill_target.dir[0] = pos_x
			skill_target.dir[1] = pos_y
			// Find the target hero along the direction
			my_pos := hero.Position()
			_find, _enemy := CheckEnemyOnDir(hero.Camp(), &my_pos, &skill_target.dir)

			if _find {
				skill_target.hero = _enemy

				skill_target.dir.Normalize()
				game.AddTarget(skill_target)
				LogStr(fmt.Sprintf("UseSkill 3, AddTarget dir skill, dir is:%v, %v", pos_x, pos_y))
			}

		} else {
			// UI mode, we're waiting for mouse to be pressed.
			hero.SetSkillTargetPos(0, 0)
			go func(hero *Vi) {
				for {
					time.Sleep(time.Duration(0.5 * float64(time.Second)))
					// Wait for left button click to select position
					skill_target_pos := hero.SkillTargetPos()
					if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
						// Use skill
						skill_target.dir[0] = skill_target_pos[0] - hero.Position()[0]
						skill_target.dir[1] = skill_target_pos[1] - hero.Position()[1]
						skill_target.dir.Normalize()

						my_pos := hero.Position()
						_find, _enemy := CheckEnemyOnDir(hero.Camp(), &my_pos, &skill_target.dir)

						if _find {
							skill_target.hero = _enemy
							callback(&skill_target)
							LogStr(fmt.Sprintf("UseSkill 3, AddTarget dir skill, dir is:%v, %v", skill_target.dir[0], skill_target.dir[1]))
						}

						break
					}
				}
			}(hero)
		}
	}

}
