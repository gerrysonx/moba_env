package main

import (
	"flag"
	"fmt"
	"image"
	"image/draw"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"./core"
	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.2/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

const windowWidth = 1000
const windowHeight = 1000

func init() {
	// GLFW event handling must run on the main OS thread
	runtime.LockOSThread()
}

func newProgram(vertexShaderSource, fragmentShaderSource string) (uint32, error) {
	vertexShader, err := compileShader(vertexShaderSource, gl.VERTEX_SHADER)
	if err != nil {
		return 0, err
	}

	fragmentShader, err := compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER)
	if err != nil {
		return 0, err
	}

	program := gl.CreateProgram()

	gl.AttachShader(program, vertexShader)
	gl.AttachShader(program, fragmentShader)
	gl.LinkProgram(program)

	var status int32
	gl.GetProgramiv(program, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetProgramiv(program, gl.INFO_LOG_LENGTH, &logLength)

		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetProgramInfoLog(program, logLength, nil, gl.Str(log))

		return 0, fmt.Errorf("failed to link program: %v", log)
	}

	gl.DeleteShader(vertexShader)
	gl.DeleteShader(fragmentShader)

	return program, nil
}

func compileShader(source string, shaderType uint32) (uint32, error) {
	shader := gl.CreateShader(shaderType)

	csources, free := gl.Strs(source)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLength)

		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetShaderInfoLog(shader, logLength, nil, gl.Str(log))

		return 0, fmt.Errorf("failed to compile %v: %v", source, log)
	}

	return shader, nil
}

func newTexture(file string) (uint32, error) {
	imgFile, err := os.Open(file)
	if err != nil {
		return 0, fmt.Errorf("texture %q not found on disk: %v", file, err)
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		return 0, err
	}

	rgba := image.NewRGBA(img.Bounds())
	if rgba.Stride != rgba.Rect.Size().X*4 {
		return 0, fmt.Errorf("unsupported stride")
	}
	draw.Draw(rgba, rgba.Bounds(), img, image.Point{0, 0}, draw.Src)

	var texture uint32
	gl.GenTextures(1, &texture)
	gl.ActiveTexture(gl.TEXTURE0)
	gl.BindTexture(gl.TEXTURE_2D, texture)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	gl.TexImage2D(
		gl.TEXTURE_2D,
		0,
		gl.RGBA,
		int32(rgba.Rect.Size().X),
		int32(rgba.Rect.Size().Y),
		0,
		gl.RGBA,
		gl.UNSIGNED_BYTE,
		gl.Ptr(rgba.Pix))

	return texture, nil
}

var vertexShader = `
#version 330
uniform mat4 projection;
uniform mat4 camera;
uniform mat4 model;
in vec3 vert;
in vec2 vertTexCoord;
out vec2 fragTexCoord;
void main() {
    fragTexCoord = vertTexCoord;
    gl_Position = projection * camera * model * vec4(vert, 1);
}
` + "\x00"

var fragmentShader = `
#version 330
uniform sampler2D tex;
uniform vec3 camp_color;

in vec2 fragTexCoord;
out vec4 outputColor;
void main() {
	outputColor = texture(tex, fragTexCoord);
	outputColor.rgb = outputColor.rgb * camp_color;
}
` + "\x00"

var cubeVertices = []float32{
	//  X, Y, Z, U, V
	// Bottom
	0, 1000.0, 0, 0.0, 1.0,
	1000.0, 1000.0, 0, 1.0, 1.0,
	1000.0, 0, 0, 1.0, 0.0,
	0, 1000.0, 0, 0.0, 1.0,
	1000.0, 0, 0, 1.0, 0.0,
	0, 0, 0, 0.0, 0.0}

func key_call_back(w *glfw.Window, char rune) {
	switch string(char) {
	case "1":
		fmt.Println("Key 1 is pressed.")
		core.GameInst.DefaultHero.UseSkill(0)

	case "2":
		fmt.Println("Key 2 is pressed.")
		core.GameInst.DefaultHero.UseSkill(1)

	case "3":
		fmt.Println("Key 3 is pressed.")
		core.GameInst.DefaultHero.UseSkill(2)

	case "4":
		fmt.Println("Key 4 is pressed.")
		core.GameInst.DefaultHero.UseSkill(3)
	}

}

func skill_rocket_grab_mouse_button_call_back(w *glfw.Window, button glfw.MouseButton, action glfw.Action, mod glfw.ModifierKey) {
	x, y := w.GetCursorPos()
	switch {
	case action == glfw.Release && button == glfw.MouseButtonLeft:

		fmt.Println("->skill_rocket_grab_mouse_button_call_back, Left mouse button is released.", button, action, mod, x, y)
		core.GameInst.DefaultHero.SetSkillTargetPos(float32(x), float32(1000-y))

	case action == glfw.Release && button == glfw.MouseButtonRight:

		fmt.Println("->skill_rocket_grab_mouse_button_call_back, Right mouse button is released.", button, action, mod, x, y)

	}

}

func mouse_button_call_back(w *glfw.Window, button glfw.MouseButton, action glfw.Action, mod glfw.ModifierKey) {
	x, y := w.GetCursorPos()
	switch {
	case action == glfw.Release && button == glfw.MouseButtonLeft:

		fmt.Println("Left mouse button is released.", button, action, mod, x, y)
		core.GameInst.DefaultHero.SetTargetPos(float32(x), float32(1000-y))

	case action == glfw.Release && button == glfw.MouseButtonRight:

		fmt.Println("Right mouse button is released.", button, action, mod, x, y)

	}
}

func main() {
	root_dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}

	//	core.InitBuffConfig("./cfg/skills.json")
	_target_frame_gap_time := flag.Float64("frame_gap", 0.03, "")
	_fix_update := flag.Bool("fix_update", true, "a bool")
	_run_render := flag.Bool("render", true, "a bool")
	_input_gap_time := flag.Float64("input_gap", 0.1, "")
	flag.Parse()

	core.GameInst = core.Game{}
	core.GameInst.Init()

	now := time.Now()
	_before_tick_time := float64(now.UnixNano()) / 1e9
	_after_tick_time := _before_tick_time
	_logic_cost_time := _before_tick_time
	_gap_time := float64(0)

	_action_stamp := int(0)
	// Set target frame time gap

	// -----------------------------render env------------------------
	if *_run_render {
		if err := glfw.Init(); err != nil {
			log.Fatalln("failed to initialize glfw:", err)
		}
		defer glfw.Terminate()

		glfw.WindowHint(glfw.Resizable, glfw.False)
		glfw.WindowHint(glfw.ContextVersionMajor, 4)
		glfw.WindowHint(glfw.ContextVersionMinor, 1)
		glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
		glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)
		window, err := glfw.CreateWindow(windowWidth, windowHeight, "Cube", nil, nil)
		if err != nil {
			panic(err)
		}
		window.MakeContextCurrent()

		// Initialize Glow
		if err := gl.Init(); err != nil {
			panic(err)
		}

		version := gl.GoStr(gl.GetString(gl.VERSION))
		fmt.Println("OpenGL version", version)

		// Configure the vertex and fragment shaders
		program, err := newProgram(vertexShader, fragmentShader)
		if err != nil {
			panic(err)
		}

		gl.UseProgram(program)

		projection := mgl32.Ident4() // mgl32.Perspective(mgl32.DegToRad(45.0), float32(windowWidth)/windowHeight, 0.1, 10.0) // mgl32.Ortho2D(-1, 1, -1, 1) //
		projectionUniform := gl.GetUniformLocation(program, gl.Str("projection\x00"))
		gl.UniformMatrix4fv(projectionUniform, 1, false, &projection[0])

		camera := mgl32.Ortho(0, 1000, 0, 1000, -1, 1) // mgl32.LookAtV(mgl32.Vec3{0, 0, -1}, mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, -1, 0})
		cameraUniform := gl.GetUniformLocation(program, gl.Str("camera\x00"))
		gl.UniformMatrix4fv(cameraUniform, 1, false, &camera[0])

		model := mgl32.Ident4()
		modelUniform := gl.GetUniformLocation(program, gl.Str("model\x00"))
		gl.UniformMatrix4fv(modelUniform, 1, false, &model[0])

		textureUniform := gl.GetUniformLocation(program, gl.Str("tex\x00"))
		gl.Uniform1i(textureUniform, 0)

		gl.BindFragDataLocation(program, 0, gl.Str("outputColor\x00"))

		// Load the texture
		full_path := fmt.Sprintf("%s/map/3_corridors.png", root_dir)
		texture, err := newTexture(full_path)
		if err != nil {
			log.Fatalln(err)
		}

		full_path = fmt.Sprintf("%s/map/footman.png", root_dir)
		tex_footman, err := newTexture(full_path)
		if err != nil {
			log.Fatalln(err)
		}

		full_path = fmt.Sprintf("%s/map/bullet.png", root_dir)
		tex_bullet, err := newTexture(full_path)
		if err != nil {
			log.Fatalln(err)
		}

		full_path = fmt.Sprintf("%s/map/hero.png", root_dir)
		tex_hero, err := newTexture(full_path)
		if err != nil {
			log.Fatalln(err)
		}

		// Configure the vertex data
		var vao uint32
		gl.GenVertexArrays(1, &vao)
		gl.BindVertexArray(vao)

		var vbo uint32
		gl.GenBuffers(1, &vbo)
		gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
		gl.BufferData(gl.ARRAY_BUFFER, len(cubeVertices)*4, gl.Ptr(cubeVertices), gl.STATIC_DRAW)

		vertAttrib := uint32(gl.GetAttribLocation(program, gl.Str("vert\x00")))
		gl.EnableVertexAttribArray(vertAttrib)
		gl.VertexAttribPointer(vertAttrib, 3, gl.FLOAT, false, 5*4, gl.PtrOffset(0))

		texCoordAttrib := uint32(gl.GetAttribLocation(program, gl.Str("vertTexCoord\x00")))
		gl.EnableVertexAttribArray(texCoordAttrib)
		gl.VertexAttribPointer(texCoordAttrib, 2, gl.FLOAT, false, 5*4, gl.PtrOffset(3*4))

		var vert_footman = []float32{
			//  X, Y, Z, U, V
			// Bottom
			0, 0, 0, 0.0, 1.0,
			0, 0, 0, 1.0, 1.0,
			0, 0, 0, 1.0, 0.0,
			0, 0, 0, 0.0, 1.0,
			0, 0, 0, 1.0, 0.0,
			0, 0, 0, 0.0, 0.0}

		var vao_footman uint32
		gl.GenVertexArrays(1, &vao_footman)
		gl.BindVertexArray(vao_footman)

		var vbo_footman uint32
		gl.GenBuffers(1, &vbo_footman)
		gl.BindBuffer(gl.ARRAY_BUFFER, vbo_footman)
		gl.BufferData(gl.ARRAY_BUFFER, len(vert_footman)*4, gl.Ptr(vert_footman), gl.STATIC_DRAW)

		vertAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vert\x00")))
		gl.EnableVertexAttribArray(vertAttrib)
		gl.VertexAttribPointer(vertAttrib, 3, gl.FLOAT, false, 5*4, gl.PtrOffset(0))

		texCoordAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vertTexCoord\x00")))
		gl.EnableVertexAttribArray(texCoordAttrib)
		gl.VertexAttribPointer(texCoordAttrib, 2, gl.FLOAT, false, 5*4, gl.PtrOffset(3*4))

		var vert_bullet = []float32{
			//  X, Y, Z, U, V
			// Bottom
			0, 0, 0, 0.0, 1.0,
			0, 0, 0, 1.0, 1.0,
			0, 0, 0, 1.0, 0.0,
			0, 0, 0, 0.0, 1.0,
			0, 0, 0, 1.0, 0.0,
			0, 0, 0, 0.0, 0.0}

		var vao_bullet uint32
		gl.GenVertexArrays(1, &vao_bullet)
		gl.BindVertexArray(vao_bullet)

		var vbo_bullet uint32
		gl.GenBuffers(1, &vbo_bullet)
		gl.BindBuffer(gl.ARRAY_BUFFER, vbo_bullet)
		gl.BufferData(gl.ARRAY_BUFFER, len(vert_bullet)*4, gl.Ptr(vert_bullet), gl.STATIC_DRAW)

		vertAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vert\x00")))
		gl.EnableVertexAttribArray(vertAttrib)
		gl.VertexAttribPointer(vertAttrib, 3, gl.FLOAT, false, 5*4, gl.PtrOffset(0))

		texCoordAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vertTexCoord\x00")))
		gl.EnableVertexAttribArray(texCoordAttrib)
		gl.VertexAttribPointer(texCoordAttrib, 2, gl.FLOAT, false, 5*4, gl.PtrOffset(3*4))

		var vert_hero = []float32{
			//  X, Y, Z, U, V
			// Bottom
			0, 0, 0, 0.0, 1.0,
			0, 0, 0, 1.0, 1.0,
			0, 0, 0, 1.0, 0.0,
			0, 0, 0, 0.0, 1.0,
			0, 0, 0, 1.0, 0.0,
			0, 0, 0, 0.0, 0.0}

		var vao_hero uint32
		gl.GenVertexArrays(1, &vao_hero)
		gl.BindVertexArray(vao_hero)

		var vbo_hero uint32
		gl.GenBuffers(1, &vbo_hero)
		gl.BindBuffer(gl.ARRAY_BUFFER, vbo_hero)
		gl.BufferData(gl.ARRAY_BUFFER, len(vert_hero)*4, gl.Ptr(vert_hero), gl.STATIC_DRAW)

		vertAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vert\x00")))
		gl.EnableVertexAttribArray(vertAttrib)
		gl.VertexAttribPointer(vertAttrib, 3, gl.FLOAT, false, 5*4, gl.PtrOffset(0))

		texCoordAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vertTexCoord\x00")))
		gl.EnableVertexAttribArray(texCoordAttrib)
		gl.VertexAttribPointer(texCoordAttrib, 2, gl.FLOAT, false, 5*4, gl.PtrOffset(3*4))

		// Configure global settings
		//gl.Enable(gl.DEPTH_TEST)
		//gl.DepthFunc(gl.LESS)
		gl.Enable(gl.BLEND)
		gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
		gl.ClearColor(1.0, 1.0, 1.0, 1.0)
		// ---------------------------------------------------------------
		core.GameInst.SetRenderParam(window, program, vao, texture, tex_footman, vao_footman, vbo_footman, vert_footman,
			tex_bullet, vao_bullet, vbo_bullet, vert_bullet,
			tex_hero, vao_hero, vbo_hero, vert_hero)

		window.SetMouseButtonCallback(mouse_button_call_back)
		window.SetCharCallback(key_call_back)
	}

	file_handle, _err := os.Create("mobacore.log")
	if _err != nil {
		fmt.Println("Create log file failed.")
		return
	}

	defer file_handle.Close()

	if *_fix_update {
		_last_input_time := core.GameInst.LogicTime
		var action_code int // action code
		for {
			// Process input from user
			if core.GameInst.LogicTime > _last_input_time+*_input_gap_time {
				// Output game state to stdout
				game_state_str := core.GameInst.DumpGameState()
				// core.LogBytes(file_handle, game_state_str)
				fmt.Printf("%d@%s\n", _action_stamp, game_state_str)
				_last_input_time = core.GameInst.LogicTime
				// Wait command from stdin
				// core.GameInst.DefaultHero.SetTargetPos(float32(x), float32(1000-y))
				//

				fmt.Scanf("%d\n", &action_code)
				_action_stamp = action_code >> 4
				action_code = action_code & 15
				battle_unit := core.GameInst.DefaultHero.(core.BaseFunc)
				cur_pos := battle_unit.Position()

				offset_x := float32(0)
				offset_y := float32(0)
				const_val := float32(100)
				switch action_code {
				case 0: // do nothing
				case 1:
					offset_x = float32(-const_val)
					offset_y = float32(-const_val)
				case 2:
					offset_x = float32(0)
					offset_y = float32(-const_val)
				case 3:
					offset_x = float32(const_val)
					offset_y = float32(-const_val)
				case 4:
					offset_x = float32(-const_val)
					offset_y = float32(0)
				case 5:
					offset_x = float32(const_val)
					offset_y = float32(0)
				case 6:
					offset_x = float32(-const_val)
					offset_y = float32(const_val)
				case 7:
					offset_x = float32(0)
					offset_y = float32(const_val)
				case 8:
					offset_x = float32(const_val)
					offset_y = float32(const_val)
				case 9:
					core.GameInst.Init()
					_last_input_time = 0
				}
				//	fmt.Printf("Set target pos:%f, %f, original:%v\n", float32(cur_pos[0]+offset_x), float32(cur_pos[1]+offset_y), cur_pos)
				core.GameInst.DefaultHero.SetTargetPos(float32(cur_pos[0]+offset_x), float32(cur_pos[1]+offset_y))
			}

			core.GameInst.Tick(*_target_frame_gap_time, *_run_render)

		}
	} else {
		for {
			_gap_time = _after_tick_time - _before_tick_time

			// fmt.Printf("------------------------------\nMain game loop, _before_tick_time is:%v\n", _before_tick_time)
			_before_tick_time = _after_tick_time
			// Do the tick
			core.GameInst.Tick(_gap_time, *_run_render)

			now = time.Now()
			_logic_cost_time = (float64(now.UnixNano()) / 1e9) - _before_tick_time

			if _logic_cost_time > *_target_frame_gap_time {
			} else {
				gap_time_in_nanoseconds := (*_target_frame_gap_time - _logic_cost_time) * float64(time.Second)
				time.Sleep(time.Duration(gap_time_in_nanoseconds))
			}
			now = time.Now()
			_after_tick_time = float64(now.UnixNano()) / 1e9
			// fmt.Printf("Main game loop, _after_tick_time is:%v, _gap_time is:%v, logic cost time is:%v\n------------------------------\n", _after_tick_time, _gap_time, _logic_cost_time)
		}
	}

}
