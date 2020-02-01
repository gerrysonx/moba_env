package core

import "github.com/ungerik/go3d/vec3"

type Creep struct {
	Hero
}

type MeleeCreep struct {
	Creep
}

func (hero *Creep) Tick(gap_time float64) {
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
	if isEnemyNearby {
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

type RangeCreep struct {
	Creep
}

//func (hero *RangeCreep) Tick(gap_time float64) {
//}

type SiegeCreep struct {
	Creep
}

//func (hero *SiegeCreep) Tick(gap_time float64) {
//}

type CreepMgr struct {
	BaseInfo
	CreepId    int32
	CreepCount int32
}

func (creepmgr *CreepMgr) Tick(gap_time float64) {
	game := &GameInst

	now_seconds := game.LogicTime
	if (creepmgr.LastAttackTime() + creepmgr.AttackFreq()) > float64(now_seconds) {
		return
	}

	// Do the spawning thing
	for i := 0; i < int(creepmgr.CreepCount); i += 1 {
		creep := HeroMgrInst.Spawn(creepmgr.CreepId, creepmgr.Camp(), creepmgr.Position()[0], creepmgr.Position()[1])
		creep.(HeroFunc).SetTargetPos(creepmgr.Direction()[0], creepmgr.Direction()[1])
		game.AddUnits = append(game.AddUnits, creep)
	}

	creepmgr.SetLastAttackTime(now_seconds)

}

func IsCreep(unit BaseFunc) bool {
	unit_id := unit.GetId()
	if unit_id < 10 && unit_id > 0 {
		return true
	}

	return false
}
