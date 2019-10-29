package core

import (
	"fmt"
	"time"

	"github.com/ungerik/go3d/vec3"
)

type Caitlyn struct {
	Hero
}

func (hero *Caitlyn) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}

var caitlyn_template Caitlyn

func (hero *Caitlyn) Init(a ...interface{}) BaseFunc {

	if caitlyn_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		caitlyn_template.InitFromJson("./cfg/caitlyn.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = caitlyn_template
	hero.SetCamp(a[0].(int32))
	hero.SetPosition(a[1].(vec3.T))
	return hero
}

func (hero *Caitlyn) UseSkill(skill_idx uint8) {

	switch skill_idx {
	case 0:
		// Piltover Peacemaker

	case 1:
		// Yordle Snap Trap

	case 2:
		// 90 Caliber Net

	case 3:
		// Ace in the Hole

	}

}
