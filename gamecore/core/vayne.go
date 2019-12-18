package core

import (
	"github.com/ungerik/go3d/vec3"
)

type Vayne struct {
	Hero
}

func (hero *Vayne) Tick(gap_time float64) {
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
		NormalAttackEnemy(hero, enemy)
	} else {
		if pos_ready {
			// Do nothing
		} else {
			Chase(hero, targetPos, gap_time)
		}
	}
}

var Vayne_template Vayne

func (hero *Vayne) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)

	if Vayne_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		Vayne_template.InitFromJson("./cfg/vayne.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = Vayne_template

	pos_x := a[1].(float32)
	pos_y := a[2].(float32)
	InitHeroWithCamp(hero, wanted_camp, pos_x, pos_y)
	hero.skillusefrequency[0] = 4
	hero.skillusefrequency[1] = 2000
	hero.skillusefrequency[2] = 4
	hero.skillusefrequency[3] = 2000
	return hero
}

func (hero *Vayne) UseSkill(skill_idx uint8, a ...interface{}) {
	game := &GameInst
	// Check CD
	now_seconds := game.LogicTime
	old_skill_use_time := hero.LastSkillUseTime(skill_idx)
	if (old_skill_use_time + hero.skillusefrequency[skill_idx]) > now_seconds {
		return
	}

	hero.SetLastSkillUseTime(skill_idx, now_seconds)

	switch skill_idx {
	case 0:

		PushEnemyAway(hero, a...)
	case 1:

	case 2:
		SlowDirEnemy(hero, a...)
		// DoStompHarm(hero)

	case 3:
		//JumpTowardsEnemy(hero, a...)
	}

	// Donot restore CD time when skill not used.
	/*
		if skill_not_used {
			hero.(BaseFunc).SetLastSkillUseTime(skill_idx, old_skill_use_time)
		}
	*/
}
