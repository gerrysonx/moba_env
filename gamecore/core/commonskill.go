package core

import (
	"fmt"
	"time"

	"github.com/ungerik/go3d/vec3"
)

type SkillTargetCallback func(*SkillTarget)
type SkillTarget struct {
	trigger_time float64
	hero         BaseFunc
	dir          vec3.T
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
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < distance) && (v.Camp() != camp) && (v.Health() > 0) {
			target_dir := unit_pos.Sub(&src_pos)
			// Check if cos(Theta)
			target_dir.Normalize()
			if vec3.Angle(target_dir, &dir) < 0.35 {
				// Means the direction is almost the same
				AddSpeedBuff([]BaseFunc{v}, 0)
				LogStr(fmt.Sprintf("SlowDirection is called, target_dir:%v", target_dir))
			}
		}
	}
}

func ChainDamage(dir vec3.T, src_pos vec3.T, camp int32, distance float32, damage float32) {
	// We shall calculate cos(Theta)
	game := &GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < distance) && (v.Camp() != camp) && (v.Health() > 0) {
			target_dir := unit_pos.Sub(&src_pos)
			// Check if cos(Theta)
			target_dir.Normalize()
			if vec3.Angle(target_dir, &dir) < 0.35 {
				// Means the direction is almost the same
				v.DealDamage(damage)
			}
		}
	}
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

func AoEDamage(src_pos vec3.T, radius float32, camp int32, damage float32) {
	// We shall calculate cos(Theta)
	game := &GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < radius) && (v.Camp() != camp) && (v.Health() > 0) {
			// Means the direction is almost the same
			v.DealDamage(damage)
		}
	}
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
			unit.SetSpeed(oldVal)
			unit.DelBuff(buff_id)
		}

		skill_target := SkillTarget{}
		skill_target.callback = callback
		skill_target.trigger_time = buff.base.Life + buff.addTime
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
