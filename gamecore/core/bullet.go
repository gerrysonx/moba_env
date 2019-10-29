package core

import (
	"github.com/ungerik/go3d/vec3"
)

type Bullet struct {
	BaseInfo
}

var bullet_template Bullet

func (bullet *Bullet) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)

	if bullet_template.Damage() == 0 {
		// Not initialized yet, initialize first, load config from json file
		bullet_template.InitFromJson("./cfg/bullet.json")
	} else {
		// Already initialized, we can copy
	}

	*bullet = bullet_template
	bullet.SetHealth(1)
	bullet.SetCamp(wanted_camp)
	bullet.SetPosition(a[1].(vec3.T))
	dir := a[2].(vec3.T)
	dir = *dir.Normalize()
	bullet.SetDirection(dir)
	bullet.SetDamage(a[3].(float32))
	game := &GameInst
	bullet.SetLastAttackTime(game.LogicTime)

	return bullet
}

func (bullet *Bullet) Tick(gap_time float64) {
	game := &GameInst

	now_seconds := game.LogicTime
	if (bullet.LastAttackTime() + bullet.AttackFreq()) < float64(now_seconds) {
		bullet.SetHealth(0)
		return
	}

	pos := bullet.Position()
	isEnemyNearby, enemy := CheckEnemyNearby(bullet.Camp(), bullet.AttackRange(), &pos)
	if isEnemyNearby {
		// Check if time to make hurt
		enemy.DealDamage(bullet.Damage())
		bullet.SetHealth(0)
	} else {
		// March towards target direction
		dir := bullet.Direction()
		dir = dir.Scaled(float32(gap_time))
		dir = dir.Scaled(float32(bullet.speed))
		newPos := vec3.Add(&pos, &dir)
		bullet.SetPosition(newPos)
	}
}
