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
	* @param t the thickness of the line
	* @res the number of points to approximate the line segment
	*/
	void drawLine(const vec3& p1, const vec3& p2, const vec3& color, size_t res = 16);

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
	* Interpolates two points on thr hyperbolic surface
	* @param p1, p2 the two points
	* @param a the ratio between the two points
	*/
	vec3 interpolate(const vec3& p1, const vec3& p2, float a);

	/*
	* Walks at a given direction from the starting point.
	* If the velocity is a unit vector, the distance of the end point
	* from the starting point will be dt.
	* @param p the starting point of the walk, its value is changed to the end point of the walk
	* @param v the velocity wich is a valid vector at p, its value is changed th the new velocity at he endpoint
	* @param dt the duration of the walk
	*/
	void walk(vec3& p, vec3& v, float dt);

	/*
	* Shifths the position and direction, as if it was on a circular orbit.
	* @param p the position of the object, its value is changed to the end point
	* @param d the direftion of the point, its value is changed to the new direction at the end point
	* @param w the angular velocity of the orbit, if positiove, the orbit is counter clockwise, otherwise it's clockwise
	* @param v the peripheral speed of the point
	* @param dt the delta time
	*/
	void orbit(vec3& p, vec3& d, float w, float v, float dt);

	// TODO comment
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

enum class ArcType {
	LEFT,
	LEFT_FORWARD,
	FORWARD,
	RIGHT_FORWARD,
	RIGHT
};

class UFOHami {
public:
	UFOHami(vec3 color) : color(color) {}

	void setMoving(bool value) { moving = value;  }

	void setMovement(ArcType value) {
		if (movement == value)
			return;

		// concatenate trail with current arc
		trail.reserve(currentArc.size());
		trail.insert(trail.end(), currentArc.begin(), currentArc.end());
		currentArc.clear();
		
		// prepare movement for the next arc
		lastPos = position;
		lastDir = direction;
		delta = 0.0f;

		movement = value;
		std::cout << "arc changed" << std::endl;
	}

	void animate(float dt) {
		animation += dt * 5.0f;

		if (moving) {
			delta += dt;

			position = lastPos;
			direction = lastDir;

			switch (movement) {
			case ArcType::FORWARD:
				HypMath::walk(position, direction, delta * VELOCITY);
				break;
			case ArcType::LEFT_FORWARD:
				HypMath::orbit(position, direction, OMEGA, VELOCITY, delta);
				break;
			case ArcType::RIGHT_FORWARD:
				HypMath::orbit(position, direction, -OMEGA, VELOCITY, delta);
				break;
			case ArcType::LEFT:
				direction = HypMath::rotateAt(lastDir, delta * OMEGA * M_PI * 2, position);
				break;
			case ArcType::RIGHT:
				direction = HypMath::rotateAt(lastDir, -delta * OMEGA * M_PI * 2, position);
				break;
			}
			HypMath::correct(position, direction);

			//TODO update trail
		}
	}

	void draw() {
		Renderer.drawCircle(position, RADIUS, color, 32);
		const float eyeRots[2] = { EYE_ANGLE, -EYE_ANGLE };
		for (auto eyeRot : eyeRots) {
			vec3 eyePos = position;
			vec3 eyeVec = HypMath::rotateAt(direction, eyeRot, position);
			HypMath::walk(eyePos, eyeVec, RADIUS);
			Renderer.drawCircle(eyePos, EYE_RADIUS, Color::white);
			Renderer.drawCircle(eyePos, EYE_RADIUS / 2.0f, Color::black);
		}
		vec3 mouthPos = position, mouthVec = direction;
		HypMath::walk(mouthPos, mouthVec, RADIUS * 0.9f);
		Renderer.drawCircle(mouthPos, MOUTH_RADIUS * sinf(animation), Color::black);
	}

	void drawTrail() {
		Renderer.drawLineStrip(trail, Color::white);
	}
private:
	vec3 position{ 0, 0, 1 }, direction{ 1, 0, 0 }, color;

	ArcType movement = ArcType::FORWARD;
	vec3 lastPos = position, lastDir = direction;
	float delta = 0.0f;
	bool moving = false;

	std::vector<vec3> trail;
	std::vector<vec3> currentArc;

	float animation = 0.0f;

	static constexpr float
		RADIUS = 0.2f,
		EYE_ANGLE = M_PI * 0.2f,
		EYE_RADIUS = 0.05f,
		MOUTH_RADIUS = 0.075f,
		
		VELOCITY = 0.75f,
		OMEGA = 0.5f;
};
/*
vec3 point = { 0.0f, 0.0f, 1.0f };
vec3 dir = { 1.0f, 0.0f, 0.0f };
*/

UFOHami
	redhami { Color::red },
	greenhami{ Color::green };

long lastTime = 0;

bool in_e = false, in_s = false, in_f = false;

// Initialization, create an OpenGL context
void onInitialization() {
	Renderer.init();
	greenhami.setMoving(true);
	greenhami.setMovement(ArcType::RIGHT_FORWARD);
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
	case 'e': in_e = true; break;
	case 's': in_s = true; break;
	case 'f': in_f = true; break;
	case 'p':
		break;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'e': in_e = false; break;
	case 's': in_s = false; break;
	case 'f': in_f = false; break;
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	const char * buttonStat = "<invalid state>";
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (!time) {
		lastTime = glutGet(GLUT_ELAPSED_TIME);
		return;
	}

	long time = glutGet(GLUT_ELAPSED_TIME);
	if (time - lastTime < 10)
		return;

	float dt = float(time - lastTime) / 1000.0f;

	bool in_rotate = in_s != in_f;
	redhami.setMoving(true); // assume movement

	if (in_e) {
		if (in_rotate) {
			if (in_s)
				redhami.setMovement(ArcType::LEFT_FORWARD);
			else
				redhami.setMovement(ArcType::RIGHT_FORWARD);
		}
		else {
			redhami.setMovement(ArcType::FORWARD);
		}
	}
	else if(in_rotate) {
		if (in_s)
			redhami.setMovement(ArcType::LEFT);
		else
			redhami.setMovement(ArcType::RIGHT); 
	}
	else {
		redhami.setMoving(false); // assumption was incorrect
	}

	//std::cout << dt << std::endl;

	/*
	HypMath::walk(point, dir, dt * input.x / 5.0f);
	dir = HypMath::rotateAt(dir, (input.y - input.z) * dt, point);
	HypMath::correct(point, dir);
	*/
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

	vec3 rdir = rv;
	vec3 last = c;
	HypMath::walk(last, rdir, r);

	for (size_t i = 0; i < res; i++) {
		angle += increment;
		rdir = HypMath::rotateAt(rv, angle, c);
		vec3 current = c;
		HypMath::walk(current, rdir, r);
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
	vp = vp / sqrtf(lorentz(vp, vp));
	// correct direction
	if (cross(v, vp).z < 0.0f)
		vp = -vp;
	return vp;
}

vec3 HypMath::rotateAt(const vec3& v, float a, const vec3& p) {
	vec3 vp = perpendicularAt(v, p);
	return cosf(a) * v + sinf(a) * vp;
}

void HypMath::walk(vec3& p, vec3& v, float dt) {
	vec3 p2 = p * coshf(dt) + v * sinhf(dt);
	v = p * sinhf(dt) + v * coshf(dt);
	p = p2;
}

void HypMath::correct(vec3& p, vec3& v) {
	p.z = sqrt(p.x * p.x + p.y * p.y + 1.0f);
	vec3 n = normalize({ p.x, p.y, -p.z });
	float e = dot(v, n);
	v = v - n * e;
	v = v / sqrtf(lorentz(v, v));
}

void HypMath::orbit(vec3& p, vec3& d, float w, float v, float dt) {
	float r = asinhf(v / w);

	vec3 centre = p;
	vec3 radius = perpendicularAt(d, p);

	// calculation the centre and radius of the orbit
	walk(centre, radius, r);
	radius = HypMath::rotateAt(-radius, w * dt, centre);

	// calculating end point of the orbit
	walk(centre, radius, r);
	p = centre;
	d = perpendicularAt(radius, p);
}
