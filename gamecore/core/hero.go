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
	Init(a ...interface{}) BaseFunc
}

type Hero struct {
	BaseInfo
	targetpos         vec3.T
	skilltargetpos    vec3.T
	skillusefrequency [4]float64
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

func (hero *Hero) ManualCtrl(gap_time float64) {
	game := &GameInst

	pos := hero.Position()
	// Check milestone distance
	targetPos := hero.TargetPos()
	dist := vec3.Distance(&pos, &targetPos)
	pos_ready := false
	if dist < 5 {
		// Already the last milestone
		pos_ready = true
	}

	isEnemyNearby, enemy := CheckEnemyNearby(hero.Camp(), hero.AttackRange(), &pos)
	if isEnemyNearby && pos_ready {
		// Check if time to make hurt
		now_seconds := game.LogicTime
		if (hero.LastAttackTime() + hero.AttackFreq()) < float64(now_seconds) {
			// Make damage
			dir_a := enemy.Position()
			dir_b := hero.Position()
			dir := vec3.Sub(&dir_a, &dir_b)
			bullet := new(Bullet).Init(hero.Camp(), hero.Position(), dir, hero.Damage())
			game.AddUnits = append(game.AddUnits, bullet)

			hero.SetLastAttackTime(now_seconds)
		}
	} else {

		if pos_ready {
			// Do nothing

		} else {
			// March towards target direction
			dir := hero.Direction()
			dir = dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)

			// Calculate new direction
			hero.direction = targetPos
			hero.direction.Sub(&newPos)
			hero.direction.Normalize()
		}
	}
}
