package core

import "github.com/ungerik/go3d/vec3"

type Tower struct {
	BaseInfo
	LastAttackee BaseFunc
}

func (tower *Tower) Tick(gap_time float64) {
	pos := tower.Position()

	if tower.LastAttackee != nil && tower.LastAttackee.Health() > 0 {
		last_attackee_pos := tower.LastAttackee.Position()
		dist := vec3.Distance(&pos, &last_attackee_pos)
		if dist < tower.AttackRange() {
			NormalAttackEnemy(tower, tower.LastAttackee)
			return
		}
	}

	isEnemyNearby, enemy := SelectFirstNonHeroEnemy(tower.Camp(), tower.AttackRange(), &pos)
	if isEnemyNearby {
		// Check if time to make hurt
		NormalAttackEnemy(tower, enemy)
		tower.LastAttackee = enemy
	} else {
		tower.LastAttackee = nil
	}
}
