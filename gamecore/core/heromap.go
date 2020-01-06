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
