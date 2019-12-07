package core

import (
	"github.com/ungerik/go3d/vec3"
)

type Vi struct {
	Hero
}

func (hero *Vi) Tick(gap_time float64) {
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
	hero.skillusefrequency[0] = 0.1
	hero.skillusefrequency[1] = 2
	hero.skillusefrequency[2] = 2
	hero.skillusefrequency[3] = 2
	return hero
}

func (hero *Vi) UseSkill(skill_idx uint8, a ...interface{}) {
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
		DoDirHarm(hero, a...)

	case 1:
		//SlowDirEnemy(hero, a...)

	case 2:
		//JumpTowardsEnemy(hero, a...)

	case 3:
		PushEnemyAway(hero, a...)
	}

}
