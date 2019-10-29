package core

import (
	"fmt"
	"time"

	"github.com/ungerik/go3d/vec3"
)

type HeroFunc interface {
	SetTargetPos(x float32, y float32)
	TargetPos() vec3.T
	SetSkillTargetPos(x float32, y float32)
	SkillTargetPos() vec3.T
	UseSkill(skill_idx uint8, a ...interface{})
}

type Hero struct {
	BaseInfo
	targetpos      vec3.T
	skilltargetpos vec3.T
}

func (baseinfo *Hero) SetTargetPos(x float32, y float32) {
	baseinfo.targetpos[0] = x
	baseinfo.targetpos[1] = y
}

func (baseinfo *Hero) TargetPos() vec3.T {
	return baseinfo.targetpos
}

func (baseinfo *Hero) SetSkillTargetPos(x float32, y float32) {
	baseinfo.skilltargetpos[0] = x
	baseinfo.skilltargetpos[1] = y
}

func (baseinfo *Hero) SkillTargetPos() vec3.T {
	return baseinfo.skilltargetpos
}

func (hero *Hero) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}

var hero_template Hero

func (hero *Hero) Init(a ...interface{}) BaseFunc {

	if hero_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		hero_template.InitFromJson("./cfg/hero.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = hero_template
	return hero
}
