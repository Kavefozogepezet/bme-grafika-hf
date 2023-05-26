//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

GPUProgram playerProg, groundProg, ropeProg;

struct VertexData {
	vec3 pos, normal;
};

float _randFloat(float value) {
	static float delta = 0.01834629f;
	static float shift = 0.82365518f;

	float r = (value + shift) / delta;
	r -= int(r);
	return r;
}

float randFloat(float value) {
	float r = value;
	for (int i = 0; i < 30; i++)
		r = _randFloat(r);
	return r;
}

template<unsigned int n>
class Noise1PF {
public:
	float A0, scale;

	Noise1PF(float amplitude, float scale) : A0(amplitude), scale(scale) {}

	VertexData operator()(float x, float z) const {
		VertexData vd;
		y = n1 = n2 = 0;

		for (int f1 = 1; f1 <= n; f1++)
			for (int f2 = 1; f2 <= n; f2++)
				component(x, f1, z, f2);
	
		vd.pos = vec3(x, y, z);
		vd.normal = normalize(-cross(vec3(1, n1, 0), vec3(0, n2, 1)));
		return vd;
	}
private:
	mutable float y, n1, n2;

	void component(float x, unsigned int f1, float z, unsigned int f2) const {
		x *= scale;
		z *= scale;

		float
			p1 = randFloat(f1) * 2 * M_PI,
			p2 = randFloat(f2) * 2 * M_PI;
		float phase =  p1 + p2;

		float A = A0 / sqrtf(f1 * f1 + f2 * f2);
		y += A * cosf(f1 * x + f2 * z + phase);

		float temp = -A * sinf(f1 * x + f2 * z + phase);
		n1 += f1 * temp;
		n2 += f2 * temp;
	}
};

struct Camera {
	vec3 position, lookat, up;
	float fov, ratio, front, back;

	Camera() :
		position(), lookat(), up(0, 1, 0),
		fov(M_PI_2), ratio(1), front(1), back(50)
	{}

	mat4 view() {
		vec3
			w = normalize(position - lookat),
			u = normalize(cross(up, w)),
			v = cross(w, u);

		mat4
			translate = TranslateMatrix(-position),
			rotate = mat4(
				u.x,	v.x,	w.x,	0,
				u.y,	v.y,	w.y,	0,
				u.z,	v.z,	w.z,	0,
				0,		0,		0,		1
			);

		return translate * rotate;
	}

	mat4 projection() {
		float
			sy = 1 / tanf(fov / 2),
			sypr = sy / ratio,
			drec = -1 / (back - front),
			fb1 = (front + back) * drec,
			fb2 = 2 * front * back * drec;

		return mat4(
			sypr,	0,		0,		0,
			0,		sy,		0,		0,
			0,		0,		fb1,	-1,
			0,		0,		fb2,	0
		);
	}
} drone, head;

class Object {
public:
	Object() :
		axis(0, 0, 1),
		inited(false)
	{}

	virtual ~Object() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}

	void draw(const mat4& VP) {
		if (!inited) {
			inited = true;
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			init();
		}

		mat4 M =
			RotationMatrix(angle, axis) *
			TranslateMatrix(position);
		mat4 MInv =
			TranslateMatrix(-position) *
			RotationMatrix(-angle, axis);
		mat4 MVP = M * VP;

		prog->Use();
		prog->setUniform(MVP, "MVP");

		glBindVertexArray(vao);
		specdraw(M, MInv);
	}
protected:
	virtual void init() = 0;
	virtual void specdraw(const mat4& M, const mat4& Minv) const = 0;
	GPUProgram* prog;

	vec3 axis, position;
	float angle;

	unsigned int vao, vbo;
	bool inited;
};

class Renderer {
public:
	void setCamera(Camera& camera) {
		currentCam = &camera;
		VP = camera.view() * camera.projection();
	}

	vec3 getCameraPos() {
		return currentCam->position;
	}

	void setDirectionalLight(vec3 dir, float intensity) {
		dirLight = normalize(dir) * intensity;
	}

	vec3 getDirectionalLight() {
		return dirLight;
	}

	void drawObject(Object& object) {
		object.draw(VP);
	}
private:
	Camera* currentCam;
	vec3 dirLight;
	mat4 VP;
} renderer;

class Rope : public Object {
public:
	void setFeetPos(vec3 pos) {
		vec3 line[2] = { vec3(), pos };
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(line), line, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, NULL, NULL);
	}
protected:
	void init() override {
		float v[6] = { 0 };

		setFeetPos(vec3());
		prog = &ropeProg;
	}

	void specdraw(const mat4& M, const mat4& MInv) const override {
		glDrawArrays(GL_LINE_STRIP, 0, 2);
	}
} rope;

class Player : public Object {
public:
	Player(Camera* head, Rope* rope) :
		headCam(head),
		rope(rope),
		jumped(false)
	{}

	void update(float dt) {
		float l = length(position);
		mat4 rot = RotationMatrix(angle, axis);
		vec4 _head = vec4(0, y / 2, 0, 1) * rot;
		vec3 head = vec3(_head.x, _head.y, _head.z);
		vec4 _up = vec4(1, 0, 0, 1) * rot;
		vec3 up = vec3(_up.x, _up.y, _up.z);

		if (l > l0) {
			vec3 Fspring = D * (l - l0) * normalize(-position);
			vec3 F = Fspring - velocity * airResistance;
			velocity = velocity + F * invMass * dt;

			float torqueSpring = dot(cross(-head, Fspring), axis);
			float torque = torqueSpring - angularv * airResistance;
			angularv += torque * invI * dt;
		}
		if (jumped)
			velocity = velocity + vec3(0, -9.81, 0) * dt;

		position = position + velocity * dt;
		angle += angularv * dt;

		headCam->position = position + head;
		headCam->lookat = position + 2 * head;
		headCam->up = up;

		rope->setFeetPos(position - head);
	}

	void jump() {
		float sec = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
		velocity.x = randFloat(sec) * 10;
		jumped = true;
	}

	vec3 getPosition() {
		return position;
	}
protected:
	void init() override {
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 36 * 6, vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, NULL);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (const void*)(sizeof(float) * 3));
		prog = &playerProg;
	}

	void specdraw(const mat4& M, const mat4& MInv) const override {
		prog->setUniform(MInv, "MInv");
		prog->setUniform(renderer.getDirectionalLight(), "light");
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}
private:
	bool jumped;
	vec3 velocity;
	float angularv;

	Camera* headCam;
	Rope* rope;
private:
	static float x, y, z, mass, invMass, I, invI, airResistance;
	static float vertices[36 * 6];

	static float l0, D;
} player(&head, &rope);

class Ground : public Object {
public:
	Ground() :
		noise(height / 2, 0.1f)
	{
		position = vec3(0, -10, 0);
	}
protected:
	void init() override {
		std::vector<VertexData> data;

		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				float u = float(j) / M;
				data.push_back(genVertex(u, float(i) / N));
				data.push_back(genVertex(u, float(i + 1) / N));
			}
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * data.size(), data.data(), GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, NULL);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (const void*)(sizeof(float) * 3));
		prog = &groundProg;
	}

	void specdraw(const mat4& M, const mat4& MInv) const override {
		prog->setUniform(M, "M");
		prog->setUniform(MInv, "MInv");
		prog->setUniform(renderer.getDirectionalLight(), "light");
		prog->setUniform(position.y - height / 2, "min");
		prog->setUniform(height, "height");
		prog->setUniform(renderer.getCameraPos(), "camPos");

		for(int i = 0; i < NStrip; i ++)
			glDrawArrays(GL_TRIANGLE_STRIP, i * MStrip, MStrip);
	}
private:
	Noise1PF<4> noise;

	VertexData genVertex(float u, float v) {
		u = (u - 0.5f) * M_PI * 20;
		v = (v - 0.5f) * M_PI * 20;
		return noise(u, v);
	}
private:
	static int N, M, NStrip, MStrip;
	static float height;
} ground;

void initShaders();

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	initShaders();
	glEnable(GL_DEPTH_TEST);
	renderer.setDirectionalLight(vec3(-1, -3, -2), 1);
}

void onDisplay() {
	glClearColor(0.5, 0.8f, 1.0f, 1.0f);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	int w = windowWidth / 2;

	glViewport(0, 0, w, windowHeight);
	renderer.setCamera(head);
	renderer.drawObject(ground);
	renderer.drawObject(rope);

	glViewport(w, 0, windowWidth - w, windowHeight);
	renderer.setCamera(drone);
	renderer.drawObject(ground);
	renderer.drawObject(rope);
	renderer.drawObject(player);

	glutSwapBuffers(); // exchange buffers for double buffering
}

void onKeyboard(unsigned char key, int pX, int pY) {
	player.jump();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void animateDrone(float dt) {
	static vec3
		h = vec3(-10, 0, 0),
		v = vec3(0, 0, -10);
	static float angle = 0.0f;

	angle += dt;
	if (angle > M_PI * 2)
		angle -= M_PI * 2;

	drone.position = h * cosf(angle) + v * sinf(angle) + vec3(0, -3, 0);
	drone.lookat = player.getPosition();
}

void onIdle() {
	static float lasttime = 0;
	float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = lasttime; t < time; t += 1 / 60.0f) {
		float dt = min(1 / 60.0f, time - t);
		animateDrone(dt);
		player.update(dt);
	}

	glutPostRedisplay();
	lasttime = time;
}

const char* const pVert = R"(
	#version 330				
	precision highp float;		

	uniform mat4 MInv, MVP;
	uniform vec3 light;

	layout(location = 0) in vec3 vp;
	layout(location = 1) in vec3 vn;
	
	out vec3 color;

	void main() {
		vec3 n = normalize((MInv * vec4(vn, 0)).xyz);
		float i = max(dot(-light, n), 0);

		color = vec3(1, 0.5, 0.5) * i;
		gl_Position = vec4(vp, 1) * MVP;
	}
)";

const char* const pFrag = R"(
	#version 330			
	precision highp float;
	
	in vec3 color;
	out vec4 outColor;

	void main() {
		outColor = vec4(color, 1);
	}
)";

const char* const rVert = R"(
	#version 330				
	precision highp float;		

	uniform mat4 MVP;		
	layout(location = 0) in vec3 vp;

	void main() {
		gl_Position = vec4(vp, 1) * MVP;
	}
)";

const char* const rFrag = R"(
	#version 330			
	precision highp float;
	
	out vec4 outColor;	

	void main() {
		outColor = vec4(1, 0, 0, 1);
	}
)";

const char* const gVert = R"(
	#version 330				
	precision highp float;		

	uniform mat4 M, MInv, MVP;
	uniform vec3 camPos;
	uniform float min, height;
		
	layout(location = 0) in vec3 vp;
	layout(location = 1) in vec3 vn;
	
	out vec3 normal, view;
	out float kd;

	void main() {
		gl_Position = vec4(vp, 1) * MVP;

		vec4 worldPos = vec4(vp, 1) * M;
		normal = (MInv * vec4(vn, 0)).xyz;
		view = camPos - worldPos.xyz / worldPos.w;
		kd = 1 - (worldPos.y - min) / height * 0.5;
	}
)";

const char* const gFrag = R"(
	#version 330			
	precision highp float;
	
	in vec3 normal, view;
	in float kd;

	out vec4 outColor;

	uniform vec3 light;

	void main() {
		vec3 n = normalize(normal);
		vec3 v = normalize(view);
		vec3 l = normalize(-light);
		vec3 h = normalize(l + v);

		float cost = max(dot(n, l), 0);
		float cosd = max(dot(n, h), 0);

		vec3 c = (vec3(0.1, 1, 0.1) * kd * cost + vec3(0.5, 0.5, 0.5) * pow(cosd, 20)) * length(light);
		outColor = vec4(c, 1);
	}
)";

void initShaders() {
	playerProg.create(pVert, pFrag, "outColor");
	ropeProg.create(rVert, rFrag, "outColor");
	groundProg.create(gVert, gFrag, "outColor");
}

float
	Player::x = 0.5,
	Player::y = 2,
	Player::z = 1,

	Player::mass = 70,
	Player::invMass = 1 / mass,
	Player::I = mass * (x * x + z * z) / 12,
	Player::invI = 1 / I,
	Player::airResistance = 5,

	Player::l0 = 2,
	Player::D = 300;

float
	Player::vertices[] = {
		-x / 2, -y / 2, z / 2,    // Bottom-left
		0, 0, 1,
		x / 2,  y / 2, z / 2,    // Top-right
		0, 0, 1,
		x / 2, -y / 2, z / 2,    // Bottom-right
		0, 0, 1,
		-x / 2, -y / 2, z / 2,    // Bottom-left
		0, 0, 1,
		-x / 2,  y / 2, z / 2,    // Top-left
		0, 0, 1,
		x / 2,  y / 2, z / 2,    // Top-right
		0, 0, 1,

		// Back face
		-x / 2, -y / 2, -z / 2,   // Bottom-left
		0, 0, -1,
		x / 2,  y / 2, -z / 2,   // Top-right
		0, 0, -1,
		x / 2, -y / 2, -z / 2,   // Bottom-right
		0, 0, -1,
		-x / 2, -y / 2, -z / 2,   // Bottom-left
		0, 0, -1,
		-x / 2,  y / 2, -z / 2,   // Top-left
		0, 0, -1,
		x / 2,  y / 2, -z / 2,   // Top-right
		0, 0, -1,

		// Left face
		-x / 2, -y / 2, -z / 2,   // Bottom-left
		-1, 0, 0,
		-x / 2,  y / 2,  z / 2,   // Top-right
		-1, 0, 0,
		-x / 2, -y / 2,  z / 2,   // Bottom-right
		-1, 0, 0,
		-x / 2, -y / 2, -z / 2,   // Bottom-left
		-1, 0, 0,
		-x / 2,  y / 2, -z / 2,   // Top-left
		-1, 0, 0,
		-x / 2,  y / 2,  z / 2,   // Top-right
		-1, 0, 0,

		// Right face
		 x / 2, -y / 2, -z / 2,   // Bottom-left
		 1, 0, 0,
		 x / 2,  y / 2,  z / 2,   // Top-right
		 1, 0, 0,
		 x / 2, -y / 2,  z / 2,   // Bottom-right
		 1, 0, 0,
		 x / 2, -y / 2, -z / 2,   // Bottom-left
		 1, 0, 0,
		 x / 2,  y / 2, -z / 2,   // Top-left
		 1, 0, 0,
		 x / 2,  y / 2,  z / 2,   // Top-right
		 1, 0, 0,

		 // Top face
		 -x / 2,  y / 2,  z / 2,   // Bottom-left
		 0, 1, 0,
		 x / 2,  y / 2, -z / 2,   // Top-right
		 0, 1, 0,
		 x / 2,  y / 2,  z / 2,   // Bottom-right
		 0, 1, 0,
		 -x / 2,  y / 2,  z / 2,   // Bottom-left
		 0, 1, 0,
		 -x / 2,  y / 2, -z / 2,   // Top-left
		 0, 1, 0,
		 x / 2,  y / 2, -z / 2,   // Top-right
		 0, 1, 0,

		 // Bottom face
		 -x / 2, -y / 2,  z / 2,   // Bottom-left
		 0, -1, 0,
		 x / 2, -y / 2, -z / 2,   // Top-right
		 0, -1, 0,
		 x / 2, -y / 2,  z / 2,   // Bottom-right
		 0, -1, 0,
		 -x / 2, -y / 2,  z / 2,   // Bottom-left
		 0, -1, 0,
		 -x / 2, -y / 2, -z / 2,    // Top-left
		 0, -1, 0,
		 x / 2, -y / 2, -z / 2,   // Top-right
		 0, -1, 0
	};

int Ground::N = 50,
	Ground::M = 50,
	Ground::NStrip = N,
	Ground::MStrip = (M + 1) * 2;

float Ground::height = 3;
