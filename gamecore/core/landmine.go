package core

import (
	"time"

	"github.com/ungerik/go3d/vec3"
)

type Landmine struct {
	BaseInfo
}

var landmine_template Landmine

func (landmine *Landmine) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)

	if landmine_template.Damage() == 0 {
		// Not initialized yet, initialize first, load config from json file
		landmine_template.InitFromJson("./cfg/landmine.json")
	} else {
		// Already initialized, we can copy
	}

	*landmine = landmine_template
	landmine.SetHealth(1)
	landmine.SetCamp(wanted_camp)
	landmine.SetPosition(a[1].(vec3.T))
	landmine.SetDirection(a[2].(vec3.T))
	now := time.Now()
	now_seconds := float64(now.UnixNano()) / 1e9
	landmine.SetLastAttackTime(now_seconds)

	return landmine
}

func (landmine *Landmine) Tick(gap_time float64) {
	now := time.Now()
	now_seconds := float64(now.UnixNano()) / 1e9
	if (landmine.LastAttackTime() + landmine.AttackFreq()) < float64(now_seconds) {
		landmine.SetHealth(0)
		return
	}

	pos := landmine.Position()
	isEnemyNearby, enemy := CheckEnemyNearby(landmine.Camp(), landmine.AttackRange(), &pos)
	if isEnemyNearby {
		// Check if time to make hurt
		enemy.DealDamage(landmine.Damage())
		landmine.SetHealth(0)
	}
}
