package core

func GetHeroByName(name string) BaseFunc {
	switch name {
	case "lusian":
		return new(Lusian)
	case "vi":
		return new(Vi)
	case "vayne":
		return new(Vayne)
	case "bullet":
		return new(Bullet)
	}

	return nil
}

func GetSkillFuncByName(name string) SkillFuncDef {
	switch name {
	case "GrabEnemyAtNose":
		return GrabEnemyAtNose
	case "PushEnemyAway":
		return PushEnemyAway
	case "DoStompHarm":
		return DoStompHarm
	case "BloodSucker":
		return BloodSucker
	case "DoDirHarm":
		return DoDirHarm
	case "SlowDirEnemy":
		return SlowDirEnemy
	case "JumpTowardsEnemy":
		return JumpTowardsEnemy
	}

	return nil
}
