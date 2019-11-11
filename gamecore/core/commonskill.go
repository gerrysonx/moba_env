package core

import (
	"time"

	"github.com/ungerik/go3d/vec3"
)

type SkillTargetCallback func(hero HeroFunc, dir vec3.T)
type SkillTarget struct {
	trigger_time float64
	hero         HeroFunc
	dir          vec3.T
	callback     SkillTargetCallback
}

func ChainDamage(dir vec3.T, src_pos vec3.T, camp int32, distance float32, damage float32) {
	// We shall calculate cos(Theta)
	game := GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		unit_pos := v.Position()
		dist = vec3.Distance(&src_pos, &unit_pos)
		if (dist < distance) && (v.Camp() != camp) && (v.Health() > 0) {
			target_dir := unit_pos.Sub(&src_pos)
			// Check if cos(Theta)
			target_dir.Normalize()
			if vec3.Dot(target_dir, &dir) > 0.9 {
				// Means the direction is almost the same
				v.DealDamage(damage)
			}
		}
	}
}

func AlterUnitPosition(dir vec3.T, move_unit BaseFunc, distance float32) {
	offset := dir.Scale(distance)
	old_pos := move_unit.Position()
	new_pos := old_pos.Add(offset)
	move_unit.SetPosition(*new_pos)
}

func AoEDamage(src_pos vec3.T, radius float32, camp int32, damage float32) {
	// We shall calculate cos(Theta)
	game := GameInst
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
	for _, v := range target_units {
		buff := NewBuff(buff_id, v.Speed())
		v.AddBuff(buff_id, buff)
		v.DealSpeed(buff.base.Val1)
		go func(target BaseFunc) {
			game := GameInst
			start_time := game.LogicTime
			for {
				if game.LogicTime > start_time+buff.base.Life {
					target.SetSpeed(buff.oldVal)
					target.DelBuff(buff_id)
					break
				}
				time.Sleep(time.Duration(10 * float64(time.Millisecond)))
			}

		}(v)
	}
}

func AddLockBuff(target_units []BaseFunc, buff_id int32) {
	for _, v := range target_units {
		buff := NewBuff(buff_id)
		v.AddBuff(buff_id, buff)
		go func(target BaseFunc) {
			game := GameInst
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
			game := GameInst
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
			game := GameInst
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
