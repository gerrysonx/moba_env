package core

import (
	"fmt"
	"time"
)

type Thresh struct {
	Hero
}

func (hero *Thresh) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}

var thresh_template Thresh

func (hero *Thresh) Init(a ...interface{}) BaseFunc {

	if thresh_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		thresh_template.InitFromJson("./cfg/thresh.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = thresh_template
	return hero
}

func (hero *Thresh) UseSkill(skill_idx uint8) {

	switch skill_idx {
	case 0:
		// Death Sentence

	case 1:
		// Dark Passage

	case 2:
		// Flay

	case 3:
		// The Box

	}

}
