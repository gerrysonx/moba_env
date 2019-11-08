package core

import (
	"encoding/json"
	"math/rand"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.2/glfw"
)

type GameTrainState struct {
	SelfHeroPosX   float32
	SelfHeroPosY   float32
	SelfHeroHealth float32
	OppoHeroPosX   float32
	OppoHeroPosY   float32
	OppoHeroHealth float32
	SelfWin        int32
}

type Game struct {
	CurrentTime     float64
	LogicTime       float64
	BattleUnits     []BaseFunc
	AddUnits        []BaseFunc
	BattleField     *BattleField
	DefaultHero     HeroFunc
	OppoHero        HeroFunc
	ManualCtrlEnemy bool
	// Render related
	window  *glfw.Window
	program uint32
	vao     uint32
	texture uint32

	tex_footman  uint32
	vao_footman  uint32
	vbo_footman  uint32
	vert_footman []float32

	tex_bullet  uint32
	vao_bullet  uint32
	vbo_bullet  uint32
	vert_bullet []float32

	tex_hero  uint32
	vao_hero  uint32
	vbo_hero  uint32
	vert_hero []float32

	train_state GameTrainState
}

func update_pos(vertice []float32, x_new float32, y_new float32, unit_width float32) {
	vertice[0] = x_new - unit_width
	vertice[1] = y_new + unit_width
	vertice[5] = x_new + unit_width
	vertice[6] = y_new + unit_width
	vertice[10] = x_new + unit_width
	vertice[11] = y_new - unit_width

	vertice[15] = x_new - unit_width
	vertice[16] = y_new + unit_width
	vertice[20] = x_new + unit_width
	vertice[21] = y_new - unit_width
	vertice[25] = x_new - unit_width
	vertice[26] = y_new - unit_width
}

func (game *Game) Testcase1() {
	var hero_pos [2][2]float32
	for {
		now := time.Now()
		rand.Seed(now.UnixNano())
		born_area_side_width := 100
		//min_player_gap := 14.0
		start_pos := (1000 - born_area_side_width) / 2
		rand_num_1 := rand.Intn(born_area_side_width)
		rand_num_2 := rand.Intn(born_area_side_width)
		rand_num_3 := rand.Intn(born_area_side_width)
		rand_num_4 := rand.Intn(born_area_side_width)
		hero_pos[0][0], hero_pos[0][1], hero_pos[1][0], hero_pos[1][1] = float32(start_pos+rand_num_1), float32(start_pos+rand_num_2), float32(start_pos+rand_num_3), float32(start_pos+rand_num_4)
		break
		/*
			if math.Pow(float64(rand_num_1-rand_num_3), 2.0)+math.Pow(float64(rand_num_2-rand_num_4), 2.0) < (min_player_gap * min_player_gap) {
				continue
			} else {
				hero_pos[0][0], hero_pos[0][1], hero_pos[1][0], hero_pos[1][1] = float32(start_pos+rand_num_1), float32(start_pos+rand_num_2), float32(start_pos+rand_num_3), float32(start_pos+rand_num_4)
				break
			}
		*/
	}
	herobase2 := new(Ezreal)
	hero1 := herobase2.Init(int32(1), hero_pos[0][0], hero_pos[0][1])
	game.BattleUnits = append(game.BattleUnits, hero1)
	game.DefaultHero = herobase2

	herobase := new(Blitzcrank)
	hero0 := herobase.Init(int32(0), hero_pos[1][0], hero_pos[1][1])
	game.BattleUnits = append(game.BattleUnits, hero0)
	game.OppoHero = herobase
}

func (game *Game) Testcase2() {
	map_units := game.BattleField.LoadMap("./map/3_corridors.png")
	game.BattleUnits = append(game.BattleUnits, map_units...)

	for i := 0; i < 1; i += 1 {
		footman := new(Footman).Init(int32(0), game.BattleField.Lanes[0][0])
		game.BattleUnits = append(game.BattleUnits, footman)
	}

	for i := 0; i < 13; i += 1 {
		footman := new(Footman).Init(int32(0), game.BattleField.Lanes[0][1])
		game.BattleUnits = append(game.BattleUnits, footman)
	}

	for i := 0; i < 10; i += 1 {
		footman := new(Footman).Init(int32(1), game.BattleField.Lanes[1][1])
		game.BattleUnits = append(game.BattleUnits, footman)
	}
}

func (game *Game) Testcase3() {
	var hero_pos [2][2]float32
	for {
		now := time.Now()
		rand.Seed(now.UnixNano())
		born_area_side_width := 100
		//min_player_gap := 14.0
		start_pos := (1000 - born_area_side_width) / 2
		rand_num_1 := rand.Intn(born_area_side_width)
		rand_num_2 := rand.Intn(born_area_side_width)
		rand_num_3 := rand.Intn(born_area_side_width)
		rand_num_4 := rand.Intn(born_area_side_width)
		hero_pos[0][0], hero_pos[0][1], hero_pos[1][0], hero_pos[1][1] = float32(start_pos+rand_num_1), float32(start_pos+rand_num_2), float32(start_pos+rand_num_3), float32(start_pos+rand_num_4)
		break
		/*
			if math.Pow(float64(rand_num_1-rand_num_3), 2.0)+math.Pow(float64(rand_num_2-rand_num_4), 2.0) < (min_player_gap * min_player_gap) {
				continue
			} else {
				hero_pos[0][0], hero_pos[0][1], hero_pos[1][0], hero_pos[1][1] = float32(start_pos+rand_num_1), float32(start_pos+rand_num_2), float32(start_pos+rand_num_3), float32(start_pos+rand_num_4)
				break
			}
		*/
	}
	herobase2 := new(Vi)
	hero1 := herobase2.Init(int32(1), hero_pos[0][0], hero_pos[0][1])
	game.BattleUnits = append(game.BattleUnits, hero1)
	game.DefaultHero = herobase2

	herobase := new(Jinx)
	hero0 := herobase.Init(int32(0), hero_pos[1][0], hero_pos[1][1])
	game.BattleUnits = append(game.BattleUnits, hero0)
	game.OppoHero = herobase
}

func (game *Game) Init() {
	game.LogicTime = 0
	game.BattleField = &BattleField{}
	// map_units := game.BattleField.LoadMap("./map/3_corridors.png")

	game.BattleUnits = []BaseFunc{}

	game.Testcase3()

}

func (game *Game) HandleInput() {

}

func (game *Game) HandleCallback() {

}

func (game *Game) GetGameState() []float32 {
	self_unit := game.DefaultHero.(BaseFunc)
	oppo_unit := game.OppoHero.(BaseFunc)

	game_state := make([]float32, 6)
	game_state[3] = self_unit.Position()[0]/1000.0 - 0.5
	game_state[4] = self_unit.Position()[1]/1000.0 - 0.5
	game_state[5] = self_unit.Health()/350.0 - 0.5

	game_state[0] = oppo_unit.Position()[0]/1000.0 - 0.5
	game_state[1] = oppo_unit.Position()[1]/1000.0 - 0.5
	game_state[2] = oppo_unit.Health()/100.0 - 0.5

	return game_state
}

func (game *Game) DumpGameState() []byte {
	//fmt.Printf("->DumpGameState, len(game.BattleUnits) is:%d, logictime:%f\n", len(game.BattleUnits), game.LogicTime)
	self_unit := game.DefaultHero.(BaseFunc)
	oppo_unit := game.OppoHero.(BaseFunc)
	if self_unit.Health() > 0 && oppo_unit.Health() > 0 {
		game.train_state.SelfWin = 0
		game.train_state.SelfHeroPosX = self_unit.Position()[0]
		game.train_state.SelfHeroPosY = self_unit.Position()[1]
		game.train_state.SelfHeroHealth = self_unit.Health()
		game.train_state.OppoHeroPosX = oppo_unit.Position()[0]
		game.train_state.OppoHeroPosY = oppo_unit.Position()[1]
		game.train_state.OppoHeroHealth = oppo_unit.Health()

	} else {
		if self_unit.Health() <= 0 {
			game.train_state.SelfWin = -1
		} else {
			game.train_state.SelfWin = 1
		}
	}

	e, err := json.Marshal(game.train_state)
	if err != nil {

	}

	return e
}

func (game *Game) Tick(gap_time float64, render bool) {
	//fmt.Printf("->gameTick, len(game.BattleUnits) is:%d, logictime:%f\n", len(game.BattleUnits), game.LogicTime)
	game.LogicTime += gap_time
	now := time.Now()
	game.CurrentTime = float64(now.UnixNano()) / 1e9
	// fmt.Printf("Game is ticking, %v-%v, gap time is:%v\n", now.Second(), game.CurrentTime, gap_time)
	var temp_arr []BaseFunc
	for _, v := range game.BattleUnits {
		if v.Health() > 0 {
			v.Tick(gap_time)
			temp_arr = append(temp_arr, v)
		} else {
			// Remove v from array, or just leave it there
		}
	}

	for _, v := range game.AddUnits {
		temp_arr = append(temp_arr, v)
	}
	game.AddUnits = game.AddUnits[:0]
	game.BattleUnits = temp_arr
	// Draw logic units on output image
	if render {
		game.Render()
	}

}

func (game *Game) SetRenderParam(window *glfw.Window, program uint32, vao uint32, texture uint32,
	tex_footman uint32, vao_footman uint32, vbo_footman uint32, vert_footman []float32,
	tex_bullet uint32, vao_bullet uint32, vbo_bullet uint32, vert_bullet []float32,
	tex_hero uint32, vao_hero uint32, vbo_hero uint32, vert_hero []float32) {
	game.window = window
	game.program = program
	game.vao = vao
	game.texture = texture
	game.tex_footman = tex_footman
	game.vao_footman = vao_footman
	game.vbo_footman = vbo_footman
	game.vert_footman = vert_footman

	game.tex_bullet = tex_bullet
	game.vao_bullet = vao_bullet
	game.vbo_bullet = vbo_bullet
	game.vert_bullet = vert_bullet

	game.tex_hero = tex_hero
	game.vao_hero = vao_hero
	game.vbo_hero = vbo_hero
	game.vert_hero = vert_hero
}

func (game *Game) render(window *glfw.Window, program uint32, vao uint32, texture uint32,
	tex_footman uint32, vao_footman uint32, vert_footman []float32,
	tex_bullet uint32, vao_bullet uint32, vert_bullet []float32,
	tex_hero uint32, vao_hero uint32, vert_hero []float32) {
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	// Update
	// time := glfw.GetTime()
	//elapsed := time - previousTime
	//previousTime = time

	//angle += elapsed
	// model = mgl32.HomogRotate3D(float32(angle), mgl32.Vec3{0, 1, 0})

	// Render
	gl.UseProgram(program)
	// gl.UniformMatrix4fv(modelUniform, 1, false, &model[0])

	// 1. Background
	colorUniform := gl.GetUniformLocation(program, gl.Str("camp_color\x00"))
	gl.Uniform3f(colorUniform, 1, 1, 1)
	gl.BindVertexArray(vao)

	gl.ActiveTexture(gl.TEXTURE0)
	gl.BindTexture(gl.TEXTURE_2D, texture)

	gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
	gl.BindVertexArray(0)
	gl.BindTexture(gl.TEXTURE_2D, 0)
	for _, v := range game.BattleUnits {
		if v.Health() > 0 {
			if f, ok := v.(*Footman); ok {
				// 2. Footman
				// 向缓冲中写入数据
				colorUniform := gl.GetUniformLocation(program, gl.Str("camp_color\x00"))
				if f.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(vert_footman, f.position[0], f.position[1], 10)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, game.vbo_footman)
				gl.BufferData(gl.ARRAY_BUFFER, len(game.vert_footman)*4, gl.Ptr(game.vert_footman), gl.STATIC_DRAW)

				gl.BindVertexArray(vao_footman)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, tex_footman)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}

			if f, ok := v.(*Bullet); ok {
				// 3. Bullet
				// 向缓冲中写入数据
				colorUniform := gl.GetUniformLocation(program, gl.Str("camp_color\x00"))
				if f.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(vert_footman, f.position[0], f.position[1], 2.5)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, game.vbo_footman)
				gl.BufferData(gl.ARRAY_BUFFER, len(game.vert_footman)*4, gl.Ptr(game.vert_footman), gl.STATIC_DRAW)

				gl.BindVertexArray(vao_footman)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, tex_footman)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}

			if _, ok := v.(HeroFunc); ok {
				// 4. Hero
				// 向缓冲中写入数据
				f0, _ := v.(BaseFunc)
				colorUniform := gl.GetUniformLocation(program, gl.Str("camp_color\x00"))
				if f0.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(vert_bullet, f0.Position()[0], f0.Position()[1], 20)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, game.vbo_bullet)
				gl.BufferData(gl.ARRAY_BUFFER, len(game.vert_bullet)*4, gl.Ptr(game.vert_bullet), gl.STATIC_DRAW)

				gl.BindVertexArray(vao_bullet)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, tex_bullet)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}

			if f, ok := v.(*Tower); ok {
				// 5. Tower
				// 向缓冲中写入数据
				colorUniform := gl.GetUniformLocation(program, gl.Str("camp_color\x00"))
				if f.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(vert_hero, f.position[0], f.position[1], 20)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, game.vbo_hero)
				gl.BufferData(gl.ARRAY_BUFFER, len(game.vert_hero)*4, gl.Ptr(game.vert_hero), gl.STATIC_DRAW)

				gl.BindVertexArray(vao_hero)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, tex_hero)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}
		}
	}

	// Maintenance
	window.SwapBuffers()
	glfw.PollEvents()
}

func (game *Game) Render() {

	game.render(game.window, game.program, game.vao, game.texture,
		game.tex_footman, game.vao_footman, game.vert_footman,
		game.tex_bullet, game.vao_bullet, game.vert_bullet,
		game.tex_hero, game.vao_hero, game.vert_hero)
	/*
		fmt.Println("->Game::Render")
		game_render_file_name := "./output/game_view.png"
		game_render_src_name := "./map/3_corridors.png"
		game_render_footman_name := "./map/footman.png"

		file_handle_1, err := os.Create(game_render_file_name)
		if err != nil {
			fmt.Println("Open file failed.", game_render_file_name)
			return
		}

		defer file_handle_1.Close()

		file_handle_2, err := os.Open(game_render_src_name)
		if err != nil {
			fmt.Println("Open file failed.", game_render_src_name)
			return
		}

		defer file_handle_2.Close()

		src_img, _ := png.Decode(file_handle_2)

		file_handle_3, err := os.Open(game_render_footman_name)
		if err != nil {
			fmt.Println("Open file failed.", game_render_footman_name)
			return
		}

		defer file_handle_3.Close()

		footman_img, _ := png.Decode(file_handle_3)

		target_img := image.NewRGBA(image.Rect(0, 0, 1000, 1000))
		draw.Draw(target_img, target_img.Bounds(), src_img, src_img.Bounds().Min, draw.Src)

		// Draw footman on target
		for _, v := range game.BattleUnits {
			if v.Health() > 0 {
				if f, ok := v.(*Footman); ok {
					draw.Draw(target_img, footman_img.Bounds().Add(image.Pt(int(f.position[0]), int(f.position[1]))), footman_img, footman_img.Bounds().Min, draw.Src)
				}
			}
		}

		png.Encode(file_handle_1, target_img)
	*/

}

// Global game object
var GameInst Game
