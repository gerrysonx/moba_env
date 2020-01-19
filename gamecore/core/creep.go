package core

type MeleeCreep struct {
	Hero
}

func (hero *MeleeCreep) Tick(gap_time float64) {
}

type RangeCreep struct {
	Hero
}

func (hero *RangeCreep) Tick(gap_time float64) {
}

type SiegeCreep struct {
	Hero
}

func (hero *SiegeCreep) Tick(gap_time float64) {
}

type CreepMgr struct {
	BaseInfo
	CreepId int32
}

func (hero *CreepMgr) Tick(gap_time float64) {
}
