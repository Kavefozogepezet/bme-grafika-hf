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

#include <stdlib.h>
#include <time.h>

GPUProgram playerProg, groundProg;

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

		prog->Use();

		mat4 M =
			RotationMatrix(angle, axis) *
			TranslateMatrix(position);
		mat4 MInv =
			TranslateMatrix(-position) *
			RotationMatrix(-angle, axis);
		mat4 MVP = M * VP;

		prog->setUniform(M, "M");
		prog->setUniform(MInv, "MInv");
		prog->setUniform(MVP, "MVP");

		glBindVertexArray(vao);
		drawcalls();
	}
protected:
	virtual void init() = 0;
	virtual void drawcalls() const = 0;
	GPUProgram* prog;

	vec3 axis, position;
	float angle;
private:
	unsigned int vao, vbo;
	bool inited;
};

class Player : public Object {
public:
	Player() :
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
	}

	void jump() {
		srand(time(NULL));
		int vx = rand() % 5000 + 5000;
		velocity.x = vx / 1000.0f;
		jumped = true;
	}

	void setHead(Camera* camera) {
		headCam = camera;
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

	void drawcalls() const override {
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}
private:
	bool jumped;
	vec3 velocity;
	float angularv;

	Camera* headCam;
private:
	static float x, y, z, mass, invMass, I, invI, airResistance;
	static float vertices[36 * 6];

	static float l0, D;
} player;

class Ground : public Object {
public:
	Ground() {
		position = vec3(0, -10, 0);
	}
protected:
	struct VertexData {
		vec3 pos, normal;
	};

	void init() override {
		std::vector<VertexData> data;

		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				float u = float(j) / M;
				data.push_back(genVertex(u, float(i) / N));
				data.push_back(genVertex(u, float(i + 1) / N));
			}
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * data.size(), data.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, NULL);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (const void*)(sizeof(float) * 3));
		prog = &playerProg;
	}

	void drawcalls() const override {
		for(int i = 0; i < NStrip; i ++)
			glDrawArrays(GL_TRIANGLE_STRIP, i * MStrip, MStrip);
	}
private:
	static int N, M, NStrip, MStrip;

	VertexData genVertex(float u, float v) {
		u = (u - 0.5f) * M_PI * 20;
		v = (v - 0.5f) * M_PI * 20;

		VertexData d = VertexData();
		d.pos = vec3(u, (sinf(u) + sinf(v)) * 0.5f, v);

		vec3 du = vec3(1, cosf(u) * 0.5f, 0);
		vec3 dv = vec3(0, cosf(v) * 0.5f, 1);
		d.normal = normalize(cross(du, dv));

		return d;
	}
} ground;

void initShaders();

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	initShaders();
	glEnable(GL_DEPTH_TEST);

	player.setHead(&head);
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	int w = windowWidth / 2;

	glViewport(0, 0, w, windowHeight);
	mat4 VP = head.view() * head.projection();
	ground.draw(VP);

	glViewport(w, 0, windowWidth - w, windowHeight);
	VP = drone.view() * drone.projection();
	player.draw(VP);
	ground.draw(VP);

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
	static long lasttime = 0;
	long time = glutGet(GLUT_ELAPSED_TIME);

	float delta = (time - lasttime) / 1000.0f;
	lasttime = time;

	animateDrone(delta);
	player.update(delta);

	glutPostRedisplay();
}

const char* const pVert = R"(
	#version 330				
	precision highp float;		

	uniform mat4 M, MInv, MVP;			
	layout(location = 0) in vec3 vp;
	layout(location = 1) in vec3 vn;
	
	out vec3 color;

	void main() {
		vec4 light = normalize(vec4(-1, 0, -2, 0));

		vec4 pos = vec4(vp, 1) * M;
		vec4 n = MInv * vec4(vn, 0);

		float i = max(dot(light, n), 0);
		color = vec3(i, i, i);

		gl_Position = vec4(vp, 1) * MVP;
	}
)";

const char* const pFrag = R"(
	#version 330			
	precision highp float;
	
	in vec3 color;
	out vec4 outColor;	

	void main() {
		outColor = vec4(color, 1) * 0.9 + vec4(0.1, 0.1, 0.1, 0);
	}
)";

void initShaders() {
	playerProg.create(pVert, pFrag, "outColor");
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

int Ground::N = 100,
	Ground::M = 100,
	Ground::NStrip = N,
	Ground::MStrip = (M + 1) * 2;
