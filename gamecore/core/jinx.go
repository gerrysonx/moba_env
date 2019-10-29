package core

import (
	"fmt"
	"time"
)

type Jinx struct {
	Hero
}

func (hero *Jinx) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}

var jinx_template Jinx

func (hero *Jinx) Init(a ...interface{}) BaseFunc {

	if jinx_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		jinx_template.InitFromJson("./cfg/jinx.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = jinx_template
	return hero
}

func (hero *Jinx) UseSkill(skill_idx uint8) {

	switch skill_idx {
	case 0:
		// Zap!

	case 1:
		// Flame Chompers!

	case 2:
		// Super Mega Death Rocket!

	}

}
