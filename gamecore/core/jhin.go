package core

import (
	"fmt"
	"time"
)

type Jhin struct {
	Hero
}

func (hero *Jhin) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}

var jhin_template Jhin

func (hero *Jhin) Init(a ...interface{}) BaseFunc {

	if jhin_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		jhin_template.InitFromJson("./cfg/jhin.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = jhin_template
	return hero
}

func (hero *Jhin) UseSkill(skill_idx uint8) {

	switch skill_idx {
	case 0:
		// Dancing Grenade

	case 1:
		// Deadly Flourish

	case 2:
		// Captive Audience

	case 3:
		// Curtain Call

	}

}
