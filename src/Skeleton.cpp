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

#include <iostream>
#include <cmath>
#include <random>
#include <time.h>

namespace Color {
	const vec3
		white{ 1, 1, 1 },
		black{ 0, 0, 0 },
		red{ 1, 0, 0 },
		green{ 0, 1, 0 },
		blue{ 0, 0, 1 };
};

class HypRenderer {
public:
	/* Initializes OpenGL, shaders, and buffers */
	void init();

	/* Clears the frame and renders the outline of the Beltrami-Poincare projection. */
	void startFrame();

	/* Swaps buffers */
	void endFrame();

	/* 
	* Draws a point on the hyperbolic surface using the Beltrami-Poincare projection.
	* @param point the point on the surface
	* @param color the color of the point
	*/
	void drawPoint(const vec3& point, const vec3& color);

	/*
	* Draws a circle on the hyperbolic surface using the Beltrami-Poincare projection.
	* @param c the centre of the circle
	* @param r the radius of the circle
	* @param color the color of the circle
	* @res the number of points to approximate the edge of the circle
	*/
	void drawCircle(const vec3& c, float r, const vec3& color, size_t res = 16);

	/*
	* Draws a line segment on the hyperbolic surface using the Beltrami-Poincare projection.
	* @param p1, p2 the two ends of the line segment
	* @param color the color of the line segment
	* @res the number of points to approximate the line segment
	*/
	void drawLine(const vec3& p1, const vec3& p2, const vec3& color, size_t res = 16);

	/*
	* Draws line segments on the hyperbolic surface using the Beltrami-Poincare projection.
	* @param points the points of the line segments
	* @param color the color of the line segment
	* @res the number of points to approximate the line segment
	*/
	void drawLineStrip(const std::vector<vec3>& points, const vec3& color);
private:
	unsigned int vao, vbo;
	GPUProgram boundary, hyperbolic;

	static float quad[18];
	static const char* boundaryVert, * boundaryFrag, * hyperbolicVert, * hyperbolicFrag;
} Renderer;

namespace HypMath {
	/*
	* Calculates the Lorentz dot product of two vectors
	* @param v1 left operand
	* @param v2 right operand
	* @returns the product
	*/
	float lorentz(const vec3& v1, const vec3& v2);

	/*
	* @param v1 A vector
	* @returns the length of the vector
	*/
	float length(const vec3& v);

	/*
	* Calculates the distance, and the direction of two points.
	* @param p, q the two points
	* @param v the function will set it to the direction of the line from p to q
	* @param d the function will set it to the distance between p and q
	*/
	void distance(const vec3& p, const vec3& q, vec3& v, float& d);

	/*
	* Moves a point in a direction with the given distance
	* @param p the point to move, its value is changed to the new point
	* @param d the direction to move the point in
	* @param l the distance between the old and new points
	*/
	void move(vec3& p, const vec3& d, float l);

	/*
	* Walks at a given direction from the starting point.
	* If the velocity is a unit vector, the distance of the end point
	* from the starting point will be dt.
	* @param p the starting point of the walk, its value is changed to the end point of the walk
	* @param d the direction wich is a valid vector at p, its value is changed th the new velocity at he endpoint 
	* @param v the walking speed
	* @param dt the duration of the walk
	*/
	void walk(vec3& p, vec3& d, float v, float dt);

	/*
	* Shifths the position and direction, as if it was on a circular orbit.
	* @param p the position of the object, its value is changed to the end point
	* @param d the direftion of the point, its value is changed to the new direction at the end point
	* @param w the angular velocity of the orbit, if positiove, the orbit is counter clockwise, otherwise it's clockwise
	* @param v the peripheral speed of the point
	* @param dt the delta time
	*/
	void orbit(vec3& p, vec3& d, float w, float v, float dt);

	/*
	* Calculates the vector, that is rotated 90 degrees.
	* @param v the vector to be rotated
	* @param p the point, where v is a valid vector
	* @return the perpendicular vector
	*/
	vec3 perpendicularAt(const vec3& v, const vec3& p);

	/*
	* Rotates the given vector along the axis defined by the normal vector of the hyperbolic surface at the point.
	* @param v the vector to be rotated
	* @param a the angle of rotation
	* @param p the point of the hyperbolic surface, where v is a valid vector
	* @returns the rotated vector
	*/
	vec3 rotateAt(const vec3& v, float a, const vec3& p);

	/*
	* Correts a point and vector,
	* so that the point will be on the hyperbolic surface,
	* and the vector will be valid at that point.
	* @param p the point to correct
	* @param v the vector to correct
	*/
	void correct(vec3& p, vec3& v);
}

/*
* Specifies the arc of the motion of a UFOHami
* LEFT, RIGHT: rotating 
* LEFT_FORWARD, RIGHT_FORWARD: circular path 
* FORWARD: straight path
*/
enum class ArcType {
	LEFT,
	LEFT_FORWARD,
	FORWARD,
	RIGHT_FORWARD,
	RIGHT,
	UNKNOWN
};

class UFOHami {
public:
	UFOHami(vec3 color) : color(color) {
		if (!initrand) {
			srand(time(NULL));
			initrand = true;
		}

		float
			randAngle = (float(rand() % 6283) * 0.01f),
			randDirAngle = (float(rand() % 6283) * 0.01f),
			randDistance = float(rand() % (SPAWN_RADIUS * 100)) * 0.01f;
		direction = HypMath::rotateAt(direction, randAngle, position);
		HypMath::walk(position, direction, 1.0f, randDistance);
		direction = HypMath::rotateAt(direction, randDirAngle, position);
		HypMath::correct(position, direction);

		lastPos = position;
		lastDir = direction;
	}

	/*
	* Sets where the UFOHami will look.
	* @param otherHami a pointer to a hami, that this hami will follow with its eyes.
	*/
	void lookingAt(UFOHami* otherHami) {
		eyeTarget = otherHami;
	}

	/*
	* @returns whether the hami eat the other hami or not
	*/
	bool eatHami(UFOHami& otherHami) {
		vec3 temp;
		float distance;
		HypMath::distance(position, otherHami.position, temp, distance);
		return distance < RADIUS * 2.0f;
	}

	/* Set wether the UFOHami is moving, or is staying still. */
	void setMoving(bool value) { moving = value;  }

	/* Set the arc of the movement the movement */
	void setMovement(ArcType value) {
		if (movement == value)
			return;
		resetArc();
		movement = value;
	}

	/*
	* The eyes and mouth of the UFOHami is animated.
	* If moving, the UFOHami moves forward on the arc of its movement.
	* @param dt the delta time that passed since the last call of this function
	*/
	void animate(float dt) {
		animation += dt * ANIMATION_SPEED;

		if (moving) {
			delta += dt;
			if (delta > MAXIMUM_ARC_LENGTH)
				resetArc();

			position = lastPos;
			direction = lastDir;

			float omega = OMEGA;

			switch (movement) {
			case ArcType::FORWARD:
				HypMath::walk(position, direction, VELOCITY, delta);
				break;
			case ArcType::RIGHT_FORWARD: omega = -OMEGA;
			case ArcType::LEFT_FORWARD:
				HypMath::orbit(position, direction, omega, VELOCITY, delta);
				break;
			case ArcType::RIGHT: omega = -OMEGA;
			case ArcType::LEFT:
				direction = HypMath::rotateAt(lastDir, delta * omega * M_PI * 2, position);
				break;
			}
			HypMath::correct(position, direction);

			//TODO update trail
		}
	}

	/* Draws the UFOHami without its trail */
	void draw() {
		Renderer.drawCircle(position, RADIUS, color, 32);

		vec3 mouthPos = position;
		HypMath::move(mouthPos, direction, RADIUS * 0.9f);
		Renderer.drawCircle(mouthPos, MOUTH_RADIUS * sinf(animation), Color::black);

		const float eyeRots[2] = { EYE_ANGLE, -EYE_ANGLE };
		for (auto eyeRot : eyeRots) {
			vec3 eyePos = position;
			vec3 eyeVec = HypMath::rotateAt(direction, eyeRot, position);
			HypMath::move(eyePos, eyeVec, RADIUS);
			Renderer.drawCircle(eyePos, EYE_RADIUS, Color::white);

			if (eyeTarget) {
				vec3 pupilDir;
				float temp;
				HypMath::distance(eyePos, eyeTarget->position, pupilDir, temp);
				HypMath::move(eyePos, pupilDir, EYE_RADIUS * 0.5f);
			}
			Renderer.drawCircle(eyePos, EYE_RADIUS * 0.5f, Color::black);
		}
	}

	/* Draws thee trail of the UFOHami. */
	void drawTrail() {
		Renderer.drawLineStrip(trail, Color::white);
		setPointsOfCurrentArc();
		Renderer.drawLineStrip(currentArc, Color::white);
	}
	vec3 position{ 0, 0, 1 }, direction{ 1, 0, 0 }, color;
private:
	ArcType movement = ArcType::FORWARD;
	vec3 lastPos = position, lastDir = direction;
	float delta = 0.0f;
	bool moving = false;

	std::vector<vec3> trail;
	std::vector<vec3> currentArc;

	float animation = 0.0f;

	UFOHami* eyeTarget = nullptr;

	/* recalculates the trail points of the current arc. */
	void setPointsOfCurrentArc() {
		float omega = OMEGA;
		float distance = delta * VELOCITY;
		size_t res = size_t(distance / TRAIL_RESOLUTION) + 1;

		currentArc.clear();
		currentArc.reserve(res);

		switch (movement) {
		case ArcType::FORWARD: {
			for (size_t i = 0; i <= res; i++) {
				vec3
					dir = lastDir,
					trailPoint = lastPos;
				float d = (distance * i) / float(res);
				HypMath::move(trailPoint, dir, d);
				currentArc.push_back(trailPoint);
			}
		} break;
		case ArcType::RIGHT_FORWARD: omega = -OMEGA;
		case ArcType::LEFT_FORWARD: {
			for (size_t i = 0; i <= res; i++) {
				vec3
					dir = lastDir,
					trailPoint = lastPos;
				float d = (delta * i) / float(res);
				HypMath::orbit(trailPoint, dir, omega, VELOCITY, d);
				currentArc.push_back(trailPoint);
			}
		} break;
		}
	}

	/*
	* Updates the starting position and direction of the
	* current arc to the current position and direction.
	* Appends the trail with the points of the current arc.
	*/
	void resetArc() {
		// concatenate trail with current arc
		trail.reserve(currentArc.size());
		trail.insert(trail.end(), currentArc.begin(), currentArc.end());
		currentArc.clear();

		// prepare movement for the next arc
		lastPos = position;
		lastDir = direction;
		delta = 0.0f;
	}
private:
	static bool initrand;

	static constexpr float
		RADIUS = 0.2f,
		EYE_ANGLE = M_PI * 0.25f,
		EYE_RADIUS = 0.075f,
		MOUTH_RADIUS = 0.075f,
		TRAIL_RESOLUTION = 0.2f,
		ANIMATION_SPEED = 3.0f,
		MAXIMUM_ARC_LENGTH = 10.0f,
		
		VELOCITY = 0.75f,
		OMEGA = 0.5f;

	static constexpr int
		SPAWN_RADIUS = 2;
}
redhami{ Color::red },
greenhami{ Color::green };

bool UFOHami::initrand = false;

/* Contains the value of GLUT_ELAPSED_TIME at the last frame */
long lastTime = 0;

/* true if redhami eat greenhami. If true, all animation must stop. */
bool gameOver = false;

/* Contains the state of the keys. Each bool field represents the corresponding key. */
struct Input {
	bool e, s, f;

	/* Returns the correct arc type based on the currents state of the keys. */
	ArcType getArcType() const;
} in;

// Initialization, create an OpenGL context
void onInitialization() {
	Renderer.init();
	greenhami.setMoving(true);
	greenhami.setMovement(ArcType::RIGHT_FORWARD);

	redhami.lookingAt(&greenhami);
	greenhami.lookingAt(&redhami);
}

// Window has become invalid: Redraw
void onDisplay() {
	Renderer.startFrame();

	greenhami.drawTrail();
	redhami.drawTrail();

	greenhami.draw();
	redhami.draw();

	Renderer.endFrame();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'e': in.e = true; break;
	case 's': in.s = true; break;
	case 'f': in.f = true; break;
	case 'p':
		vec3 dirrot = HypMath::rotateAt(redhami.direction, 0.1f, redhami.position);
		float angle = acos(HypMath::lorentz(dirrot, redhami.direction) / (HypMath::length(dirrot) * HypMath::length(redhami.direction)));
		//std::cout << "angle: " << angle << std::endl;
		break;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'e': in.e = false; break;
	case 's': in.s = false; break;
	case 'f': in.f = false; break;
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (gameOver)
		return;

	if (!time) {
		lastTime = glutGet(GLUT_ELAPSED_TIME);
		return;
	}

	long time = glutGet(GLUT_ELAPSED_TIME);
	if (time - lastTime < 10)
		return;

	if (redhami.eatHami(greenhami))
		gameOver = true;

	float dt = float(time - lastTime) / 1000.0f;
	auto arc = in.getArcType();

	redhami.setMoving(false); // assume the hami stopped moving
	if (arc != ArcType::UNKNOWN) {
		redhami.setMovement(arc);
		redhami.setMoving(true);
	}
	
	redhami.animate(dt);
	greenhami.animate(dt);

	lastTime = time;

	glutPostRedisplay();
}

const char
* HypRenderer::boundaryVert = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec3 vp;
	out vec2 radius;

	void main() {
		radius = vec2(vp.xy);
		gl_Position = vec4(vp.x, vp.y, 0, 1);
	}
)",
* HypRenderer::boundaryFrag = R"(
	#version 330
	precision highp float;
	
	in vec2 radius;
	out vec4 outColor;

	void main() {
		if(length(radius) > 1.0)
			outColor = vec4(0.1f, 0.1, 0.1, 1);
		else
			outColor = vec4(0, 0, 0, 0);
	}
)",
* HypRenderer::hyperbolicVert = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec3 vp;

	void main() {
		float a = 1 / (vp.z + 1);
		vec3 proj = vec3(0, 0, -1) * (1 - a) + vp * a;
		gl_Position = vec4(proj.x, proj.y, 0, 1);
	}
)",
* HypRenderer::hyperbolicFrag = R"(
	#version 330
	precision highp float;
	
	uniform vec3 color;
	out vec4 outColor;

	void main() {
		outColor = vec4(color, 1);
	}
)";

void HypRenderer::init() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		3, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU
	boundary.create(boundaryVert, boundaryFrag, "outColor");
	hyperbolic.create(hyperbolicVert, hyperbolicFrag, "outColor");
}

void HypRenderer::startFrame() {
	boundary.Use();
	glBindVertexArray(vao);  // Draw call
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(quad),  // # bytes
		quad,	      	// address
		GL_DYNAMIC_DRAW);	// we do not change later
	glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 6/*# Elements*/);
	hyperbolic.Use();
}

void HypRenderer::endFrame() {
	glutSwapBuffers();
}

void HypRenderer::drawPoint(const vec3& point, const vec3& color) {
	int location = glGetUniformLocation(hyperbolic.getId(), "color");
	glUniform3f(location, color.x, color.y, color.z); // 3 floats
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vec3),  // # bytes
		&point.x,	      	// address
		GL_DYNAMIC_DRAW);	// we do not change later
	glDrawArrays(GL_POINTS, 0, 1);
}

void HypRenderer::drawCircle(const vec3& c, float r, const vec3& color, size_t res) {
	float angle = 0;
	float increment = 2 * M_PI / float(res);

	std::vector<vec3> vertices {};
	vertices.reserve(res * 3);

	float y;
	vec3 rv;

	if (c.x == 0.0f) {
		rv = { 0.0f, 1.0f, 0.0f};
	} else {
		float y = sqrtf(c.x * c.x / (c.y * c.y + c.x * c.x));
		rv = { -c.y * y / c.x, y, 0.0f };
	}

	vec3 r1 = { 1, 0, 0 }, r2 = { 0, 1, 0 };

	vec3 last = c;
	HypMath::move(last, rv, r);

	for (size_t i = 0; i < res; i++) {
		angle += increment;
		vec3 rdir = HypMath::rotateAt(rv, angle, c);
		vec3 current = c;
		HypMath::move(current, rdir, r);
		vertices.push_back(c);
		vertices.push_back(last);
		vertices.push_back(current);
		last = current;
	}

	int location = glGetUniformLocation(hyperbolic.getId(), "color");
	glUniform3f(location, color.x, color.y, color.z); // 3 floats
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vec3) * vertices.size(),  // # bytes
		vertices.data(),	      	// address
		GL_DYNAMIC_DRAW);	// we do not change later
	glDrawArrays(GL_TRIANGLES, 0, res * 3);
}

void HypRenderer::drawLineStrip(const std::vector<vec3>& points, const vec3& color) {
	int location = glGetUniformLocation(hyperbolic.getId(), "color");
	glUniform3f(location, color.x, color.y, color.z); // 3 floats
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vec3) * points.size(),  // # bytes
		points.data(),	      	// address
		GL_DYNAMIC_DRAW);	// we do not change later
	glDrawArrays(GL_LINE_STRIP, 0, points.size());
}

float HypRenderer::quad[18] = {
	1.0f, 1.0f, 0.0f,		-1.0f, 1.0f, 0.0f,		1.0f, -1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f,		1.0f, -1.0f, 0.0f,		-1.0f, -1.0f, 0.0f,
};

float HypMath::lorentz(const vec3& v1, const vec3& v2) {
	return v1.x * v2.x + v1.y * v2.y - v1.z * v2.z;
}

float HypMath::length(const vec3& v) {
	return sqrtf(lorentz(v, v));
}

void HypMath::distance(const vec3& p, const vec3& q, vec3& v, float& d) {
	d = acoshf(-lorentz(q, p));
	v = (q - p * coshf(d)) / sinhf(d);
}

vec3 HypMath::perpendicularAt(const vec3& v, const vec3& p) {
	vec3 vp;
	float expr1 = v.z * p.x - p.z * v.x;
	// check if we can divide with expr1, no need for a bias
	if (expr1 == 0.0f) {
		float z = p.x / p.z;
		vp = { 1.0f, 0.0f, z };
	} else {
		float
			x = (p.z * v.y - v.z * p.y) / expr1,
			z = (p.x * x + p.y) / p.z;
		vp = { x, 1.0f, z };
	}
	vp = vp / HypMath::length(vp);
	// correct direction
	if (cross(v, vp).z < 0.0f)
		vp = -vp;
	return vp;
}

vec3 HypMath::rotateAt(const vec3& v, float a, const vec3& p) {
	vec3 vp = perpendicularAt(v, p);
	return cosf(a) * v + sinf(a) * vp;
}

void HypMath::move(vec3& p, const vec3& d, float l) {
	p = p * coshf(l) + d * sinhf(l);
}

void HypMath::walk(vec3& p, vec3& d, float v, float dt) {
	float l = v * dt;
	vec3 p2 = p;
	move(p2, d, l);
	d = p * sinhf(l) + d * coshf(l);
	p = p2;
}

void HypMath::correct(vec3& p, vec3& v) {
	p.z = sqrt(p.x * p.x + p.y * p.y + 1.0f);
	vec3 n = normalize({ p.x, p.y, -p.z });
	float e = dot(v, n);
	v = v - n * e;
	v = v / HypMath::length(v);
}

void HypMath::orbit(vec3& p, vec3& d, float w, float v, float dt) {
	float r = asinhf(v / w);

	vec3 centre = p;
	vec3 radius = perpendicularAt(d, p);

	// calculation the centre and radius of the orbit
	walk(centre, radius, 1.0f, r);
	radius = HypMath::rotateAt(-radius, w * dt, centre);

	// calculating end point of the orbit
	walk(centre, radius, 1.0f, r);
	p = centre;
	d = perpendicularAt(radius, p);
}

ArcType Input::getArcType() const {
	bool in_rotate = s != f;
	if (e)
		if (in_rotate)
			if (s)
				return ArcType::LEFT_FORWARD;
			else
				return ArcType::RIGHT_FORWARD;
		else
			return ArcType::FORWARD;
	else if (in_rotate)
		if (s)
			return ArcType::LEFT;
		else
			return ArcType::RIGHT;
	return ArcType::UNKNOWN;
}
