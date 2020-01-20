package core

type Tower struct {
	BaseInfo
}

func (hero *Tower) Tick(gap_time float64) {
	pos := hero.Position()

	isEnemyNearby, enemy := CheckEnemyNearby(hero.Camp(), hero.AttackRange(), &pos)
	if isEnemyNearby {
		// Check if time to make hurt
		NormalAttackEnemy(hero, enemy)
	}
}
