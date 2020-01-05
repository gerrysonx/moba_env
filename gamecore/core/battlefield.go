package core

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/ungerik/go3d/vec3"
)

type BattleField struct {
	CurrentTime  float64
	Map          image.Image
	Bounds       image.Rectangle
	Width        int32
	Height       int32
	Lanes        [2][3][]vec3.T
	Restricted_x float32
	Restricted_y float32
	Restricted_w float32
	Restricted_h float32
}

func (battle_field *BattleField) Within(pos_x float32, pos_y float32) bool {
	if pos_x > battle_field.Restricted_x &&
		pos_x < battle_field.Restricted_x+battle_field.Restricted_w &&
		pos_y > battle_field.Restricted_y &&
		pos_y < battle_field.Restricted_y+battle_field.Restricted_h {
		return true
	}

	return false
}

func (battle_field *BattleField) LoadMap(filename string) []BaseFunc {
	// Init lanes
	// Camp 0
	// Upper lane
	battle_field.Lanes[0][0] = []vec3.T{{280, 20, 0}, {980, 20, 0}, {980, 980, 0}} //[]vec3.T{{16, 32, 0}, {526, 20, 0}, {980, 19, 0}}
	// Middle lane
	battle_field.Lanes[0][1] = []vec3.T{{20, 20, 0}, {408, 511, 0}, {980, 980, 0}}
	// Lower lane
	battle_field.Lanes[0][2] = []vec3.T{{20, 20, 0}, {20, 980, 0}, {980, 980, 0}}

	// Camp 1
	// Upper lane
	battle_field.Lanes[1][0] = []vec3.T{{983, 983, 0}, {980, 20, 0}, {19, 19, 0}}
	// Middle lane
	battle_field.Lanes[1][1] = []vec3.T{{974, 983, 0}, {408, 511, 0}, {19, 19, 0}}
	// Lower lane
	battle_field.Lanes[1][2] = []vec3.T{{974, 983, 0}, {470, 985, 0}, {19, 19, 0}}

	file_handle, err := os.Open(filename)
	if err != nil {
		fmt.Printf("Open file failed:%v", filename)
		return nil
	}

	defer file_handle.Close()
	img, img_type, err := image.Decode(file_handle)
	if err != nil {
		fmt.Printf("Open file failed:%v, type:%v", filename, img_type)
		return nil
	}

	battle_field.Map = img
	/*
		x := 399
		y := 18
		color := img.At(x, y)

		r, g, b, _ := color.RGBA()

		fmt.Printf("color is:%v", color)
	*/

	battle_field.Bounds = img.Bounds()
	battle_field.Width = int32(battle_field.Bounds.Max.X - battle_field.Bounds.Min.X)
	battle_field.Height = int32(battle_field.Bounds.Max.Y - battle_field.Bounds.Min.Y)

	battle_units := []BaseFunc{}
	// Mini clustering
	for idx := int32(0); idx < battle_field.Width; idx += 1 {
		for idy := int32(0); idy < battle_field.Height; idy += 1 {
			color := img.At(int(idx), int(idy))
			r, g, b, _ := color.RGBA()
			var unit BaseFunc
			unit_camp := -1
			unit_id := -1
			switch {
			case r == 0 && g != 0 && b == 0:
				unit_camp = 0
				unit_id = UnitTypeBullet

			case r != 0 && g == 0 && b == 0:
				unit_camp = 1
				unit_id = UnitTypeBullet

			case r == 0 && g != 0 && b != 0:
				unit_camp = 1
				unit_id = UnitTypeAncient

			case r != 0 && g == 0 && b != 0:
				unit_camp = 0
				unit_id = UnitTypeAncient

			case r == 0 && g == 0 && b != 0:
				unit_camp = 2
				unit_id = UnitTypeMonster
			default:
				continue
			}

			has_cluster_core := false
			var pos vec3.T
			pos[0], pos[1] = float32(idx), float32(idy)
			for _, tmp_unit := range battle_units {
				if unit.Camp() != tmp_unit.Camp() {
					continue
				}
				tmp_pos := tmp_unit.Position()

				distance := vec3.Distance(&pos, &tmp_pos)
				if distance < 40 {
					has_cluster_core = true
					break
				}

			}
			if !has_cluster_core {
				unit = HeroMgrInst.Spawn(unit_id, int32(unit_camp), float32(idx), float32(idy))
				battle_units = append(battle_units, unit)
			}
		}
	}
	fmt.Printf("Loaded %v units.", len(battle_units))
	return battle_units
}
