package core

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/ungerik/go3d/vec3"
)

type SkillTargetCallback func(*SkillTarget)
type SkillTarget struct {
	trigger_time float64
	src          BaseFunc
	hero         BaseFunc
	dir          vec3.T
	pos          vec3.T
	val0         float32
	val1         int32
	buff         *Buff
	callback     SkillTargetCallback
}

func SlowDirection(dir vec3.T, src_pos vec3.T, camp int32, distance float32) {
	// We shall calculate cos(Theta)
	game := &GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < distance) && (v.Camp() != camp) && (v.Health() > 0) {
			target_dir := unit_pos.Sub(&src_pos)
			// Check if cos(Theta)
			target_dir.Normalize()
			if vec3.Angle(target_dir, &dir) < 0.35 {
				// Means the direction is almost the same
				AddSpeedBuff([]BaseFunc{v}, BuffSpeedSlow)
				LogStr(fmt.Sprintf("SlowDirection is called, target_dir:%v", target_dir))
			}
		}
	}
}

func ChainDamage(dir vec3.T, src_pos vec3.T, camp int32, distance float32, damage float32) (bool, []BaseFunc) {
	// We shall calculate cos(Theta)
	var targets []BaseFunc
	damaged_dealed := false
	game := &GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < distance) && (v.Camp() != camp) && (v.Health() > 0) {
			target_dir := unit_pos.Sub(&src_pos)
			// Check if cos(Theta)
			target_dir.Normalize()
			if vec3.Angle(target_dir, &dir) < 0.35 {
				// Means the direction is almost the same
				targets = append(targets, v)
				v.DealDamage(damage)
				damaged_dealed = true
			}
		}
	}

	return damaged_dealed, targets
}

func AlterUnitPosition(dir vec3.T, move_unit BaseFunc, distance float32) {
	game := &GameInst

	offset := dir.Scale(distance)
	old_pos := move_unit.Position()
	new_pos := old_pos.Add(offset)
	// Check if new_pos is within restriction area, if not shall fail
	is_target_within := game.BattleField.Within(new_pos[0], new_pos[1])
	if is_target_within {
		move_unit.SetPosition(*new_pos)
	}
}

func AoESucker(src_hero BaseFunc, radius float32, camp int32, damage float32, revive_hp float32) (bool, []BaseFunc) {
	src_pos := src_hero.Position()
	// We shall calculate cos(Theta)
	damaged_dealed := false
	var hit_units []BaseFunc
	game := &GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < radius) && (v.Camp() != camp) && (v.Health() > 0) {
			// Means the direction is almost the same
			LogStr(fmt.Sprintf("AoEDamage toward enemy:%v, attack range:%v, dist:%v, radius:%v, time:%v",
				v.GetId(), v.AttackRange(), dist, radius, game.LogicTime))

			v.DealDamage(damage)
			hit_units = append(hit_units, v)
			damaged_dealed = true
		}
	}

	if damaged_dealed {
		src_hero.DealDamage(-revive_hp)
	}

	return damaged_dealed, hit_units
}

func AoEDamage(src_pos vec3.T, radius float32, camp int32, damage float32) (bool, []BaseFunc) {
	// We shall calculate cos(Theta)
	damaged_dealed := false
	var hit_units []BaseFunc
	game := &GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < radius) && (v.Camp() != camp) && (v.Health() > 0) {
			// Means the direction is almost the same
			LogStr(fmt.Sprintf("AoEDamage toward enemy:%v, attack range:%v, dist:%v, radius:%v, time:%v",
				v.GetId(), v.AttackRange(), dist, radius, game.LogicTime))

			v.DealDamage(damage)
			hit_units = append(hit_units, v)
			damaged_dealed = true
		}
	}

	return damaged_dealed, hit_units
}

func AddSpeedBuff(target_units []BaseFunc, buff_id int32) {
	game := &GameInst
	for _, v := range target_units {
		buff := NewBuff(buff_id, v.Speed())
		v.AddBuff(buff_id, buff)
		v.DealSpeed(buff.base.Val1)
		// Add callback
		callback := func(skill_target *SkillTarget) {
			var unit BaseFunc
			var oldVal = skill_target.buff.oldVal
			var buff_id = skill_target.buff.base.Id
			unit = skill_target.hero
			speed_before_change := v.Speed()
			unit.SetSpeed(oldVal)
			unit.DelBuff(buff_id)
			LogStr(fmt.Sprintf("SlowDirection reset is called, buff_id:%v, speed_before_change:%v, after:%v", buff_id, speed_before_change, oldVal))
		}

		skill_target := SkillTarget{}
		skill_target.callback = callback
		skill_target.trigger_time = buff.base.Life + buff.addTime
		LogStr(fmt.Sprintf("SlowDirection AddSpeedBuff is called, trigger_time:%v, buff.base.Life:%v, now speed:%v", skill_target.trigger_time, buff.base.Life, v.Speed()))
		skill_target.buff = buff
		skill_target.hero = v
		game.AddTarget(skill_target)
	}
}

func AddLockBuff(target_units []BaseFunc, buff_id int32) {
	for _, v := range target_units {
		buff := NewBuff(buff_id)
		v.AddBuff(buff_id, buff)
		go func(target BaseFunc) {
			game := &GameInst
			start_time := game.LogicTime
			for {
				if game.LogicTime > start_time+buff.base.Life {
					target.DelBuff(buff_id)
					break
				}
				time.Sleep(time.Duration(10 * float64(time.Millisecond)))
			}
		}(v)
	}
}

func AddDamageProofBuff(target_units []BaseFunc, buff_id int32) {
	for _, v := range target_units {
		buff := NewBuff(buff_id)
		v.AddBuff(buff_id, buff)
		go func(target BaseFunc) {
			game := &GameInst
			start_time := game.LogicTime
			for {
				if game.LogicTime > start_time+buff.base.Life {
					target.DelBuff(buff_id)
					break
				}
				time.Sleep(time.Duration(10 * float64(time.Millisecond)))
			}
		}(v)
	}
}

func AddAttackFreqBuff(target_units []BaseFunc, buff_id int32) {
	for _, v := range target_units {
		buff := NewBuff(buff_id)
		v.AddBuff(buff_id, buff)
		v.SetAttackFreq(float64(buff.base.Val1))
		go func(target BaseFunc) {
			game := &GameInst
			start_time := game.LogicTime
			for {
				if game.LogicTime > start_time+buff.base.Life {
					target.SetAttackFreq(float64(buff.oldVal))
					target.DelBuff(buff_id)
					break
				}
				time.Sleep(time.Duration(10 * float64(time.Millisecond)))
			}
		}(v)
	}
}

func GrabEnemyAtNose(hero HeroFunc, a ...interface{}) {
	game := &GameInst

	has_more_params := len(a) > 0

	callback := func(skill_target *SkillTarget) {
		var unit BaseFunc

		unit = skill_target.hero
		pos := skill_target.pos
		now := time.Now()
		rand.Seed(now.UnixNano())
		pos[0] += rand.Float32() - 0.5
		pos[1] += rand.Float32() - 0.5
		unit.SetPosition(pos)
		LogStr(fmt.Sprintf("GrabEnemyAtNose harm is called, has_more_params:%v, now_seconds:%v", has_more_params, game.LogicTime))
	}

	LogStr(fmt.Sprintf("UseSkill GrabEnemyAtNose is called, has_more_params:%v, now_seconds:%v", has_more_params, game.LogicTime))

	skill_target := SkillTarget{}
	skill_target.callback = callback
	skill_target.trigger_time = 0

	if has_more_params {
		pos_x := a[0].(float32)
		pos_y := a[1].(float32)

		skill_target.dir[0] = pos_x
		skill_target.dir[1] = pos_y
		// Find the target hero along the direction
		my_pos := hero.(BaseFunc).Position()
		_find, _enemy := CheckEnemyOnDirWithinDist(hero.(BaseFunc).Camp(), &my_pos, &skill_target.dir, 50)

		if _find {
			skill_target.hero = _enemy
			skill_target.pos = my_pos
			skill_target.dir.Normalize()
			game.AddTarget(skill_target)
			LogStr(fmt.Sprintf("UseSkill GrabEnemyAtNose, AddTarget dir skill, dir is:%v, %v", pos_x, pos_y))
		}

	} else {
		// UI mode, we're waiting for mouse to be pressed.
		hero.SetSkillTargetPos(0, 0)
		go func(hero HeroFunc) {
			for {
				time.Sleep(time.Duration(0.5 * float64(time.Second)))
				// Wait for left button click to select position
				skill_target_pos := hero.SkillTargetPos()
				if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
					// Use skill
					skill_target.dir[0] = skill_target_pos[0] - hero.(BaseFunc).Position()[0]
					skill_target.dir[1] = skill_target_pos[1] - hero.(BaseFunc).Position()[1]
					skill_target.dir.Normalize()

					my_pos := hero.(BaseFunc).Position()
					_find, _enemy := CheckEnemyOnDir(hero.(BaseFunc).Camp(), &my_pos, &skill_target.dir)

					if _find {
						skill_target.hero = _enemy
						skill_target.pos = my_pos
						callback(&skill_target)
						LogStr(fmt.Sprintf("UseSkill 3, AddTarget dir skill, dir is:%v, %v", skill_target.dir[0], skill_target.dir[1]))
					}

					break
				}
			}
		}(hero)
	}
}

func PushEnemyAway(hero HeroFunc, a ...interface{}) {
	game := &GameInst

	has_more_params := len(a) > 0

	callback := func(skill_target *SkillTarget) {
		var unit BaseFunc
		var dir vec3.T
		var src BaseFunc
		unit = skill_target.hero
		dir = skill_target.dir
		src = skill_target.src
		AlterUnitPosition(dir, unit, 40)
		LogStr(fmt.Sprintf("PushEnemyAway is emitted, from :%v to %v, time:%v", src.GetId(), unit.GetId(), game.LogicTime))
	}

	LogStr(fmt.Sprintf("PushEnemyAway is called, has_more_params:%v, now_seconds:%v", has_more_params, game.LogicTime))

	skill_target := SkillTarget{}
	skill_target.callback = callback
	skill_target.trigger_time = 0

	if has_more_params {
		pos_x := a[0].(float32)
		pos_y := a[1].(float32)

		skill_target.dir[0] = pos_x
		skill_target.dir[1] = pos_y
		// Find the target hero along the direction
		my_pos := hero.(BaseFunc).Position()
		_find, _enemy := CheckEnemyOnDirWithinDist(hero.(BaseFunc).Camp(), &my_pos, &skill_target.dir, 60)

		if _find {
			skill_target.hero = _enemy
			skill_target.src = hero.(BaseFunc)
			skill_target.dir.Normalize()
			callback(&skill_target)
			//	game.AddTarget(skill_target)
		}

	} else {
		// UI mode, we're waiting for mouse to be pressed.
		hero.SetSkillTargetPos(0, 0)
		go func(hero HeroFunc) {
			for {
				time.Sleep(time.Duration(0.5 * float64(time.Second)))
				// Wait for left button click to select position
				skill_target_pos := hero.SkillTargetPos()
				if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
					// Use skill
					skill_target.dir[0] = skill_target_pos[0] - hero.(BaseFunc).Position()[0]
					skill_target.dir[1] = skill_target_pos[1] - hero.(BaseFunc).Position()[1]
					skill_target.dir.Normalize()

					my_pos := hero.(BaseFunc).Position()
					_find, _enemy := CheckEnemyOnDir(hero.(BaseFunc).Camp(), &my_pos, &skill_target.dir)

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

func DoStompHarm(hero HeroFunc, a ...interface{}) {
	game := &GameInst
	damage_dealed, targets := AoEDamage(hero.(BaseFunc).Position(), 20, hero.(BaseFunc).Camp(), 500.0)
	if damage_dealed {
		LogStr(fmt.Sprintf("DoStompHarm emit real harm.from:%v to:%v, at time:%v", hero.(BaseFunc).GetId(), targets[0].GetId(), game.LogicTime))
	}
}

func BloodSucker(hero HeroFunc, a ...interface{}) {
	game := &GameInst
	damage_dealed, targets := AoESucker(hero.(BaseFunc), 20, hero.(BaseFunc).Camp(), 400.0, 400.0)
	if damage_dealed {
		LogStr(fmt.Sprintf("BloodSucker emit real harm.from:%v to:%v, at time:%v", hero.(BaseFunc).GetId(), targets[0].GetId(), game.LogicTime))
	}
}

func DoDirHarm(hero HeroFunc, a ...interface{}) {
	game := &GameInst
	// Clear skilltarget pos
	// Check if has more parameters
	has_more_params := len(a) > 0

	callback := func(skill_target *SkillTarget) {
		var unit BaseFunc
		var dir vec3.T
		unit = skill_target.hero
		dir = skill_target.dir
		damage_dealed, targets := ChainDamage(dir, unit.Position(), unit.Camp(), 20, 500.0)
		if damage_dealed {
			LogStr(fmt.Sprintf("DoDirHarm emit real harm, from:%v to:%v, time:%v", unit.GetId(), targets[0].GetId(), game.LogicTime))
		}

	}

	LogStr(fmt.Sprintf("DoBigHarm is called, has_more_params:%v, now_seconds:%v", has_more_params, game.LogicTime))

	skill_target := SkillTarget{}
	skill_target.callback = callback
	skill_target.trigger_time = 0

	if has_more_params {
		pos_x := a[0].(float32)
		pos_y := a[1].(float32)

		skill_target.hero = hero.(BaseFunc)
		skill_target.dir[0] = pos_x
		skill_target.dir[1] = pos_y
		skill_target.dir.Normalize()
		callback(&skill_target)
		//	game.AddTarget(skill_target)
	} else {
		// UI mode, we're waiting for mouse to be pressed.
		hero.SetSkillTargetPos(0, 0)
		go func(hero HeroFunc) {
			for {
				time.Sleep(time.Duration(0.5 * float64(time.Second)))
				// Wait for left button click to select position
				skill_target_pos := hero.SkillTargetPos()
				if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
					// Use skill
					skill_target.dir[0] = skill_target_pos[0] - hero.(BaseFunc).Position()[0]
					skill_target.dir[1] = skill_target_pos[1] - hero.(BaseFunc).Position()[1]
					skill_target.dir.Normalize()
					skill_target.hero = hero.(BaseFunc)
					callback(&skill_target)
					break
				}
			}
		}(hero)
	}
}

func SlowDirEnemy(hero HeroFunc, a ...interface{}) {
	game := &GameInst
	has_more_params := len(a) > 0

	skill_dist := float32(100.0)
	callback := func(skill_target *SkillTarget) {
		var unit BaseFunc
		unit = skill_target.hero
		AddSpeedBuff([]BaseFunc{unit}, BuffSpeedSlow)
		LogStr(fmt.Sprintf("SlowDirection is called, AddSpeedBuff calling finished."))
	}

	LogStr(fmt.Sprintf("SlowDirEnemy is called, has_more_params:%v", has_more_params))

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
		my_pos := hero.(BaseFunc).Position()
		_find, _enemy := CheckEnemyOnDirWithinDist(hero.(BaseFunc).Camp(), &my_pos, &skill_target.dir, skill_dist)

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
		go func(hero HeroFunc) {
			for {
				time.Sleep(time.Duration(0.5 * float64(time.Second)))
				// Wait for left button click to select position
				skill_target_pos := hero.SkillTargetPos()
				if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
					// Use skill
					skill_target.dir[0] = skill_target_pos[0] - hero.(BaseFunc).Position()[0]
					skill_target.dir[1] = skill_target_pos[1] - hero.(BaseFunc).Position()[1]
					skill_target.dir.Normalize()

					my_pos := hero.(BaseFunc).Position()
					_find, _enemy := CheckEnemyOnDirWithinDist(hero.(BaseFunc).Camp(), &my_pos, &skill_target.dir, skill_dist)

					if _find {
						skill_target.hero = _enemy

						callback(&skill_target)
					}
					break
				}
			}
		}(hero)
	}
}

func JumpTowardsEnemy(hero HeroFunc, a ...interface{}) {
	game := &GameInst
	// Clear skilltarget pos
	// Check if has more parameters
	has_more_params := len(a) > 0

	callback := func(skill_target *SkillTarget) {
		var unit BaseFunc
		var dir vec3.T
		unit = skill_target.hero
		dir = skill_target.dir
		AlterUnitPosition(dir, unit, 60)
	}

	LogStr(fmt.Sprintf("UseSkill 2 is called, has_more_params:%v.", has_more_params))

	skill_target := SkillTarget{}
	skill_target.callback = callback
	skill_target.trigger_time = 0

	if has_more_params {
		pos_x := a[0].(float32)
		pos_y := a[1].(float32)

		skill_target.hero = hero.(BaseFunc)
		skill_target.dir[0] = pos_x
		skill_target.dir[1] = pos_y
		skill_target.dir.Normalize()
		game.AddTarget(skill_target)
		LogStr(fmt.Sprintf("UseSkill 2, AddTarget dir skill, dir is:%v, %v", pos_x, pos_y))
	} else {
		// UI mode, we're waiting for mouse to be pressed.
		hero.SetSkillTargetPos(0, 0)
		go func(hero HeroFunc) {
			for {
				time.Sleep(time.Duration(0.5 * float64(time.Second)))
				// Wait for left button click to select position
				skill_target_pos := hero.SkillTargetPos()
				if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
					// Use skill
					skill_target.dir[0] = skill_target_pos[0] - hero.(BaseFunc).Position()[0]
					skill_target.dir[1] = skill_target_pos[1] - hero.(BaseFunc).Position()[1]
					skill_target.dir.Normalize()
					skill_target.hero = hero.(BaseFunc)
					callback(&skill_target)
					break
				}
			}
		}(hero)
	}
}

func NormalAttackEnemy(hero BaseFunc, enemy BaseFunc) {
	game := &GameInst
	now_seconds := game.LogicTime
	if (hero.LastAttackTime() + hero.AttackFreq()) < float64(now_seconds) {
		// Make damage
		dir_a := enemy.Position()
		dir_b := hero.Position()
		dir := vec3.Sub(&dir_a, &dir_b)
		bullet := HeroMgrInst.Spawn(int32(UnitTypeBullet), hero.Camp(), hero.Position()[0], hero.Position()[1])
		dir.Normalize()
		bullet.SetDirection(dir)
		bullet.SetDamage(hero.Damage())
		bullet.SetLastAttackTime(game.LogicTime)

		game.AddUnits = append(game.AddUnits, bullet)

		hero.SetLastAttackTime(now_seconds)
	}
}

func Chase(hero BaseFunc, pos_enemy vec3.T, gap_time float64) {
	pos := hero.Position()

	// Check milestone distance
	targetPos := pos_enemy
	dist := vec3.Distance(&pos, &targetPos)
	if dist < 4 {

	} else {
		// March towards enemy pos
		dir := hero.Direction()
		dir = dir.Scaled(float32(gap_time))
		dir = dir.Scaled(float32(hero.Speed()))
		newPos := vec3.Add(&pos, &dir)
		hero.SetPosition(newPos)

		// Calculate new direction
		dir = targetPos
		dir.Sub(&newPos)
		dir.Normalize()
		hero.SetDirection(dir)
	}

}

func Escape(hero BaseFunc, pos_enemy vec3.T, gap_time float64) {
	pos := hero.Position()

	// Escape from enemy pos
	// Check milestone distance
	targetPos := pos_enemy
	dist := vec3.Distance(&pos, &targetPos)
	if dist > 100 {
		// Already the last milestone
		var dir vec3.T
		dir[0] = 0
		dir[1] = 0
		hero.SetDirection(dir)
	} else {
		dir := hero.Direction()
		dir = dir.Scaled(float32(gap_time))
		dir = dir.Scaled(float32(hero.Speed()))
		newPos := vec3.Add(&pos, &dir)
		hero.SetPosition(newPos)

		// Calculate new direction
		dir = targetPos
		dir.Sub(&newPos)
		dir.Normalize()
		dir[0] = -dir[0]
		dir[1] = -dir[1]
		hero.SetDirection(dir)
	}
}
