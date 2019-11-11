package render

import (
	"fmt"
	"image"
	"image/draw"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"../core"
	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.2/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

type Renderer struct {
	// Render related
	window  *glfw.Window
	program uint32
	vao     uint32
	texture uint32
	vbo     uint32

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
	game      *core.Game
}

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

func mouse_button_call_back(w *glfw.Window, button glfw.MouseButton, action glfw.Action, mod glfw.ModifierKey) {
	x, y := w.GetCursorPos()
	switch {
	case action == glfw.Release && button == glfw.MouseButtonLeft:

		//	fmt.Println("Left mouse button is released.", button, action, mod, x, y)
		if core.GameInst.ManualCtrlEnemy {
			core.GameInst.OppoHero.SetTargetPos(float32(x), float32(1000-y))
		} else {
			core.GameInst.DefaultHero.SetTargetPos(float32(x), float32(1000-y))
		}

	case action == glfw.Release && button == glfw.MouseButtonRight:

		core.GameInst.DefaultHero.SetSkillTargetPos(float32(x), float32(1000-y))

	}
}
func (renderer *Renderer) Release() {
	glfw.Terminate()
}

func (renderer *Renderer) InitRenderEnv(game *core.Game) {
	renderer.game = game

	root_dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}

	if err := glfw.Init(); err != nil {
		log.Fatalln("failed to initialize glfw:", err)
	}

	glfw.WindowHint(glfw.Resizable, glfw.False)
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)
	window, err := glfw.CreateWindow(windowWidth, windowHeight, "Cube", nil, nil)
	if err != nil {
		panic(err)
	}
	renderer.window = window
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
	renderer.program = program

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
	renderer.texture = texture

	full_path = fmt.Sprintf("%s/map/footman.png", root_dir)
	tex_footman, err := newTexture(full_path)
	if err != nil {
		log.Fatalln(err)
	}
	renderer.tex_footman = tex_footman

	full_path = fmt.Sprintf("%s/map/bullet.png", root_dir)
	tex_bullet, err := newTexture(full_path)
	if err != nil {
		log.Fatalln(err)
	}
	renderer.tex_bullet = tex_bullet

	full_path = fmt.Sprintf("%s/map/hero.png", root_dir)
	tex_hero, err := newTexture(full_path)
	if err != nil {
		log.Fatalln(err)
	}
	renderer.tex_hero = tex_hero

	// Configure the vertex data
	var vao uint32
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)
	renderer.vao = vao

	var vbo uint32
	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(cubeVertices)*4, gl.Ptr(cubeVertices), gl.STATIC_DRAW)
	renderer.vbo = vbo

	vertAttrib := uint32(gl.GetAttribLocation(program, gl.Str("vert\x00")))
	gl.EnableVertexAttribArray(vertAttrib)
	gl.VertexAttribPointer(vertAttrib, 3, gl.FLOAT, false, 5*4, gl.PtrOffset(0))

	texCoordAttrib := uint32(gl.GetAttribLocation(program, gl.Str("vertTexCoord\x00")))
	gl.EnableVertexAttribArray(texCoordAttrib)
	gl.VertexAttribPointer(texCoordAttrib, 2, gl.FLOAT, false, 5*4, gl.PtrOffset(3*4))

	renderer.vert_footman = []float32{
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
	renderer.vao_footman = vao_footman

	var vbo_footman uint32
	gl.GenBuffers(1, &vbo_footman)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo_footman)
	gl.BufferData(gl.ARRAY_BUFFER, len(renderer.vert_footman)*4, gl.Ptr(renderer.vert_footman), gl.STATIC_DRAW)
	renderer.vbo_footman = vbo_footman

	vertAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vert\x00")))
	gl.EnableVertexAttribArray(vertAttrib)
	gl.VertexAttribPointer(vertAttrib, 3, gl.FLOAT, false, 5*4, gl.PtrOffset(0))

	texCoordAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vertTexCoord\x00")))
	gl.EnableVertexAttribArray(texCoordAttrib)
	gl.VertexAttribPointer(texCoordAttrib, 2, gl.FLOAT, false, 5*4, gl.PtrOffset(3*4))

	renderer.vert_bullet = []float32{
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
	renderer.vao_bullet = vao_bullet

	var vbo_bullet uint32
	gl.GenBuffers(1, &vbo_bullet)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo_bullet)
	gl.BufferData(gl.ARRAY_BUFFER, len(renderer.vert_bullet)*4, gl.Ptr(renderer.vert_bullet), gl.STATIC_DRAW)
	renderer.vbo_bullet = vbo_bullet

	vertAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vert\x00")))
	gl.EnableVertexAttribArray(vertAttrib)
	gl.VertexAttribPointer(vertAttrib, 3, gl.FLOAT, false, 5*4, gl.PtrOffset(0))

	texCoordAttrib = uint32(gl.GetAttribLocation(program, gl.Str("vertTexCoord\x00")))
	gl.EnableVertexAttribArray(texCoordAttrib)
	gl.VertexAttribPointer(texCoordAttrib, 2, gl.FLOAT, false, 5*4, gl.PtrOffset(3*4))

	renderer.vert_hero = []float32{
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
	renderer.vao_hero = vao_hero

	var vbo_hero uint32
	gl.GenBuffers(1, &vbo_hero)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo_hero)
	gl.BufferData(gl.ARRAY_BUFFER, len(renderer.vert_hero)*4, gl.Ptr(renderer.vert_hero), gl.STATIC_DRAW)
	renderer.vbo_hero = vbo_hero

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
	// core.GameInst.SetRenderParam(window, program, vao, texture, tex_footman, vao_footman, vbo_footman, vert_footman,
	//	tex_bullet, vao_bullet, vbo_bullet, vert_bullet,
	//	tex_hero, vao_hero, vbo_hero, vert_hero)

	window.SetMouseButtonCallback(mouse_button_call_back)
	window.SetCharCallback(key_call_back)
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

func (renderer *Renderer) Render() {

	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	// Render
	gl.UseProgram(renderer.program)
	// gl.UniformMatrix4fv(modelUniform, 1, false, &model[0])

	// 1. Background
	colorUniform := gl.GetUniformLocation(renderer.program, gl.Str("camp_color\x00"))
	gl.Uniform3f(colorUniform, 1, 1, 1)
	gl.BindVertexArray(renderer.vao)

	gl.ActiveTexture(gl.TEXTURE0)
	gl.BindTexture(gl.TEXTURE_2D, renderer.texture)

	gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
	gl.BindVertexArray(0)
	gl.BindTexture(gl.TEXTURE_2D, 0)
	for _, v := range renderer.game.BattleUnits {
		if v.Health() > 0 {
			if f, ok := v.(*core.Footman); ok {
				// 2. Footman
				// 向缓冲中写入数据
				colorUniform := gl.GetUniformLocation(renderer.program, gl.Str("camp_color\x00"))
				if f.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(renderer.vert_footman, f.Position()[0], f.Position()[1], 10)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, renderer.vbo_footman)
				gl.BufferData(gl.ARRAY_BUFFER, len(renderer.vert_footman)*4, gl.Ptr(renderer.vert_footman), gl.STATIC_DRAW)

				gl.BindVertexArray(renderer.vao_footman)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, renderer.tex_footman)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}

			if f, ok := v.(*core.Bullet); ok {
				// 3. Bullet
				// 向缓冲中写入数据
				colorUniform := gl.GetUniformLocation(renderer.program, gl.Str("camp_color\x00"))
				if f.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(renderer.vert_footman, f.Position()[0], f.Position()[1], 2.5)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, renderer.vbo_footman)
				gl.BufferData(gl.ARRAY_BUFFER, len(renderer.vert_footman)*4, gl.Ptr(renderer.vert_footman), gl.STATIC_DRAW)

				gl.BindVertexArray(renderer.vao_footman)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, renderer.tex_footman)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}

			if _, ok := v.(core.HeroFunc); ok {
				// 4. Hero
				// 向缓冲中写入数据
				f0, _ := v.(core.BaseFunc)
				colorUniform := gl.GetUniformLocation(renderer.program, gl.Str("camp_color\x00"))
				if f0.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(renderer.vert_bullet, f0.Position()[0], f0.Position()[1], 20)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, renderer.vbo_bullet)
				gl.BufferData(gl.ARRAY_BUFFER, len(renderer.vert_bullet)*4, gl.Ptr(renderer.vert_bullet), gl.STATIC_DRAW)

				gl.BindVertexArray(renderer.vao_bullet)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, renderer.tex_bullet)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}

			if f, ok := v.(*core.Tower); ok {
				// 5. Tower
				// 向缓冲中写入数据
				colorUniform := gl.GetUniformLocation(renderer.program, gl.Str("camp_color\x00"))
				if f.Camp() == 0 {
					gl.Uniform3f(colorUniform, 0, 0.8, 0)
				} else {
					gl.Uniform3f(colorUniform, 0.8, 0, 0)
				}

				update_pos(renderer.vert_hero, f.Position()[0], f.Position()[1], 20)
				// 完成够别忘了告诉OpenGL我们不再需要它了
				gl.BindBuffer(gl.ARRAY_BUFFER, renderer.vbo_hero)
				gl.BufferData(gl.ARRAY_BUFFER, len(renderer.vert_hero)*4, gl.Ptr(renderer.vert_hero), gl.STATIC_DRAW)

				gl.BindVertexArray(renderer.vao_hero)

				gl.ActiveTexture(gl.TEXTURE0)
				gl.BindTexture(gl.TEXTURE_2D, renderer.tex_hero)

				gl.DrawArrays(gl.TRIANGLES, 0, 1*2*3)
				continue
			}
		}
	}

	// Maintenance
	renderer.window.SwapBuffers()
	glfw.PollEvents()
}

var RendererInst Renderer
