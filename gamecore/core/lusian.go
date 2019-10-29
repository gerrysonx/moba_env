package core

import (
	"fmt"
	"time"
)

type Lusian struct {
	Hero
}

func (hero *Lusian) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}

var lusian_template Lusian

func (hero *Lusian) Init(a ...interface{}) BaseFunc {

	if lusian_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		lusian_template.InitFromJson("./cfg/lusian.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = lusian_template
	return hero
}

func (hero *Lusian) UseSkill(skill_idx uint8) {

	switch skill_idx {
	case 0:
		// Piercing Light

	case 1:
		// Ardent Blaze

	case 2:
		// Relentless Pursuit

	case 3:
		// The Culling

	}

}
