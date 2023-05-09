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

#include <array>
#include <vector>

constexpr float EPSILON = 0.01f;

struct Ray {
	vec3 start, direction;

	vec3 operator()(float t) const {
		return start + t * direction;
	}
};

struct Hit {
	vec3 position, normal;
	float t = -1;

	operator bool() {
		return t >= 0;
	}
};

struct Object {
	vec3 position;
	virtual Hit intersect(const Ray& ray) const = 0;
};

Object* objects[6] = { nullptr };
enum ObjectIndexes {
	RED_DETECTOR, GREEN_DETECTOR, BLUE_DETECTOR, ROOM, OCTAHEDRON, DODECAHEDRON
};

class Mesh : public Object {
public:
	using Face = std::array<uint32_t, 3>;
public:
	Mesh() {}

	Mesh(
		std::initializer_list<Face> faces,
		std::initializer_list<vec3> vertices) :
		faces(faces), vertices(vertices)
	{
		radius = length(this->vertices[0] - position);
		for (auto& v : this->vertices) {
			float temp = length(v - position);
			if (temp < radius)
				radius = temp;
		}
		radius += EPSILON;
	}

	Hit intersect(const Ray& ray) const override {

		vec3 dist = ray.start - position;
		float
			a = dot(ray.direction, ray.direction),
			b = dot(dist, ray.direction) * 2,
			c = dot(dist, dist) - radius * radius,
			discr = b * b - 4 * a * c;

		if (discr < 0)
			return {};

		Hit hit;
		for (auto& face : faces) {
			vec3
				r0 = vertices[face[0]] + position,
				r1 = vertices[face[1]] + position,
				r2 = vertices[face[2]] + position,
				n = cross(r1 - r0, r2 - r0);

			if (dot(ray.direction, n) > 0)
				continue;

			float t = dot(n, (r0 - ray.start)) / dot(ray.direction, n);
			if (t < 0)
				continue;
			vec3 p = ray(t);

			if (dot(n, cross(r1 - r0, p - r0)) < 0
				|| dot(n, cross(r2 - r1, p - r1)) < 0
				|| dot(n, cross(r0 - r2, p - r2)) < 0)
				continue;

			if (!hit || hit.t > t)
				hit = { p, normalize(n), t };
		}
		return hit;
	}
private:
	std::vector<vec3> vertices;
	std::vector<Face> faces;
	float radius;
};

class Detector : public Object {
public:
	Detector() {}

	Detector(vec3 position, vec3 direction, float angle, float height, vec3 color) :
		angle(angle * 0.5f), height(height), position(position), direction(normalize(direction)), color(color)
	{}

	Hit intersect(const Ray& ray) const override {
		vec3 dist = ray.start - position;

		float
			rd_d = dot(ray.direction, direction),
			dist_d = dot(dist, direction),
			dist2 = dot(dist, dist),
			rd_dist = dot(ray.direction, dist),
			cos2a = cosf(angle);
		cos2a *= cos2a;

		float a = rd_d * rd_d - cos2a;
		float b = 2 * (rd_d * dist_d - rd_dist * cos2a);
		float c = dist_d * dist_d - dist2 * cos2a;

		float discr = b * b - 4 * a * c;
		if (discr < 0)
			return {};

		discr = sqrt(discr);
		float ts[2] = {
			(-b - discr) / (2 * a),
			(-b + discr) / (2 * a)
		};

		if (ts[0] < 0 && ts[1] < 0)
			return {};

		if (ts[0] > ts[1]) {
			float temp = ts[0];
			ts[0] = ts[1];
			ts[1] = temp;
		}
	
		for (float t : ts) {
			vec3 p = ray(t) - position;
			float h = dot(p, direction);

			if (t >= 0 && h <= height && h >= 0) {
				vec3 n = normalize(p * dot(direction, p) / dot(p, p) - direction);
				if (dot(n, dist) < 0)
					n = -n;
				return { ray(t), n, t };
			}
		}
		return {};
	}

	bool can_detect(vec3 point) const {
		vec3 view = normalize(point - light_pos());
		float a = acosf(dot(view, direction));
		return a <= angle;
	}

	float get_distance(vec3 point) const {
		return length(light_pos() - point);
	}

	Ray ray(vec3 point) {
		return { point, normalize(light_pos() - point) };
	}

	const vec3& get_color() { return color; }

	void place(Hit h) {
		position = h.position;
		direction = h.normal;
	}
private:
	float angle, height;
	vec3 position, direction, color;

	vec3 light_pos() const {
		return position + direction * EPSILON;
	}
};

Mesh meshes[3];
Detector detectors[3];

class Camera {
public:
	void set(vec3 position, vec3 look_at, vec3 up, float field_of_view) {
		p = position;
		la = look_at;
		fov = field_of_view;

		vec3 w = p - la;
		float win_size = length(w) * tanf(fov / 2);
		r = normalize(cross(up, w)) * win_size;
		u = normalize(cross(w, r)) * win_size;
	}

	Ray ray(int x, int y) {
		vec3 dir = la - p
			+ r * (2 * (x + 0.5f) / windowWidth - 1)
			+ u * (2 * (y + 0.5f) / windowHeight - 1); 
		return { p, normalize(dir) };
	}

	vec3 direction() const {
		return normalize(la - p);
	}
private:
	vec3 p, la, r, u;
	float fov;
} camera;

Hit cast_ray(const Ray& ray) {
	Hit hit;
	for (auto obj_ptr : objects) {
		if (obj_ptr == nullptr)
			continue;

		Hit temp = obj_ptr->intersect(ray);
		if (!hit || (temp && temp.t < hit.t))
			hit = temp;
	}
	return hit;
}

void initOpenGL();
void initObjects();
void displayFrame(const std::vector<vec4>& data, int width, int height);

void onInitialization() {
	initOpenGL();
	initObjects();
}

void onDisplay() {
	std::vector<vec4> data(windowWidth * windowHeight);
	for (int y = 0; y < windowWidth; y++) {
		for (int x = 0; x < windowHeight; x++) {
			Ray r = camera.ray(x, y);
			Hit h = cast_ray(r);
			if (!h)
				continue;

			float L = 0.2 * (1 + dot(-h.normal, camera.direction()));
			vec3 rgb { L, L, L };
			vec3 hitp = h.position + h.normal * EPSILON;

			for (auto& d : detectors) {
				if (!d.can_detect(hitp))
					continue;
				
				Ray dr = d.ray(hitp);
				Hit dh = cast_ray(dr);

				if (!dh)
					continue;

				float hd = length(dh.position - hitp);
				float dd = d.get_distance(hitp);
				if(hd < dd)
					continue;

				rgb = rgb + d.get_color() * 1 / (1 + dd * dd);
				
			}
			data[y * windowHeight + x] = vec4(rgb.x, rgb.y, rgb.z, 1);
		}
	}
	displayFrame(data, windowWidth, windowHeight);
}

void onKeyboard(unsigned char key, int pX, int pY) {}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {
	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
		Ray r = camera.ray(pX, windowHeight - pY);
		Hit h = cast_ray(r);

		Detector* closest = nullptr;
		for (auto& d : detectors) {
			if (closest == nullptr || closest->get_distance(h.position) > d.get_distance(h.position))
				closest = &d;
		}

		closest->place(h);
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
unsigned int texture;

const char* const vertexSource = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec2 vp;
	out vec2 tc;

	void main() {
		tc = (vp + vec2(1, 1)) / 2;
		gl_Position = vec4(vp.x, vp.y, 0, 1);
	}
)";

const char* const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform sampler2D frame;
	in vec2 tc;
	out vec4 outColor;

	void main() {
		outColor = texture(frame, tc);
	}
)";

void initOpenGL() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	unsigned int vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	float vertices[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(vertices),
		vertices,
		GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0,
		2, GL_FLOAT, GL_FALSE,
		0, NULL);

	gpuProgram.create(vertexSource, fragmentSource, "outColor");

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void displayFrame(const std::vector<vec4>& data, int width, int height) {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	int texLoc = glGetUniformLocation(gpuProgram.getId(), "frame");
	glUniform1i(texLoc, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
		GL_RGBA, GL_FLOAT, data.data());

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glutSwapBuffers();
}

void initObjects() {
	camera.set({ -3.5, 0, -5.5 }, { 0, 0, 0 }, { 0, 1, 0 }, M_PI / 3);

	meshes[0] = {
		{
			{ 4, 0, 2 },
			{ 4, 2, 1 },
			{ 4, 1, 3 },
			{ 4, 3, 0 },
			{ 5, 2, 0 },
			{ 5, 1, 2 },
			{ 5, 3, 1 },
			{ 5, 0, 3 },
		},
		{
			{ 1.0, 0.0, 0.0 },
			{ -1.0, 0.0, 0.0 },
			{ 0.0, 1.0, 0.0 },
			{ 0.0, -1.0, 0.0 },
			{ 0.0, 0.0, 1.0 },
			{ 0.0, 0.0, -1.0 },
		}
	};
	meshes[0].position = {-1, -1, 0};
	objects[OCTAHEDRON] = meshes;

	meshes[1] = {
		{
			{ 8, 16, 0 },
			{ 8, 9, 4 },
			{ 12, 8, 0 },
			{ 12, 13, 1 },
			{ 16, 12, 0 },
			{ 16, 17, 2 },
			{ 1, 9, 8 },
			{ 1, 18, 5 },
			{ 2, 13, 12 },
			{ 2, 10, 3 },
			{ 4, 17, 16 },
			{ 4, 14, 6 },
			{ 5, 4, 9 },
			{ 5, 15, 14 },
			{ 11, 17, 6 },
			{ 11, 10, 2 },
			{ 19, 13, 3 },
			{ 19, 18, 1 },
			{ 7, 18, 19 },
			{ 15, 5, 18 },
			{ 7, 14, 15 },
			{ 11, 6, 14 },
			{ 7, 10, 11 },
			{ 19, 3, 10 },
			{ 8, 4, 16 },
			{ 12, 1, 8 },
			{ 16, 2, 12 },
			{ 1, 5, 9 },
			{ 2, 3, 13 },
			{ 4, 6, 17 },
			{ 5, 14, 4 },
			{ 11, 2, 17 },
			{ 19, 1, 13 },
			{ 7, 15, 18 },
			{ 7, 11, 14 },
			{ 7, 19, 10 },
		},
		{
			{ 0.5773502588272095, 0.5773502588272095, 0.5773502588272095 },
			{ 0.5773502588272095, 0.5773502588272095, -0.5773502588272095 },
			{ 0.5773502588272095, -0.5773502588272095, 0.5773502588272095 },
			{ 0.5773502588272095, -0.5773502588272095, -0.5773502588272095 },
			{ -0.5773502588272095, 0.5773502588272095, 0.5773502588272095 },
			{ -0.5773502588272095, 0.5773502588272095, -0.5773502588272095 },
			{ -0.5773502588272095, -0.5773502588272095, 0.5773502588272095 },
			{ -0.5773502588272095, -0.5773502588272095, -0.5773502588272095 },
			{ 0.35682210326194763, 0.9341723322868347, 0.0 },
			{ -0.35682210326194763, 0.9341723322868347, 0.0 },
			{ 0.35682210326194763, -0.9341723322868347, 0.0 },
			{ -0.35682210326194763, -0.9341723322868347, 0.0 },
			{ 0.9341723322868347, 0.0, 0.35682210326194763 },
			{ 0.9341723322868347, 0.0, -0.35682210326194763 },
			{ -0.9341723322868347, 0.0, 0.35682210326194763 },
			{ -0.9341723322868347, 0.0, -0.35682210326194763 },
			{ 0.0, 0.35682210326194763, 0.9341723322868347 },
			{ 0.0, -0.35682210326194763, 0.9341723322868347 },
			{ 0.0, 0.35682210326194763, -0.9341723322868347 },
			{ 0.0, -0.35682210326194763, -0.9341723322868347 },
		}
	};
	meshes[1].position = {1, -1, 0};
	objects[DODECAHEDRON] = meshes + 1;

	meshes[2] = {
		{
			{ 2, 1, 0 },
			{ 6, 3, 2 },
			{ 4, 7, 6 },
			{ 0, 5, 4 },
			{ 0, 6, 2 },
			{ 5, 3, 7 },
			{ 2, 3, 1 },
			{ 6, 7, 3 },
			{ 4, 5, 7 },
			{ 0, 1, 5 },
			{ 0, 4, 6 },
			{ 5, 1, 3 },
		},
		{
			{ -2.0, -2.0, -2.0 },
			{ -2.0, -2.0, 2.0 },
			{ -2.0, 2.0, -2.0 },
			{ -2.0, 2.0, 2.0 },
			{ 2.0, -2.0, -2.0 },
			{ 2.0, -2.0, 2.0 },
			{ 2.0, 2.0, -2.0 },
			{ 2.0, 2.0, 2.0 },
		}
	};
	objects[ROOM] = meshes + 2;
	
	detectors[0] = Detector({ 2, 1.54325, -0.42163 }, { -1, 0, 0 }, M_PI / 4, 0.5f, { 1, 0, 0 });
	detectors[1] = Detector({ 0.130424, -1.10452, -0.271748 }, { -0.850651, -0.525731, 0 }, M_PI / 4, 0.5f, { 0, 1, 0 });
	detectors[2] = Detector({ -0.441254, 2, -0.306689 }, { 0, -1, 0 }, M_PI / 4, 0.5f, { 0, 0, 1 });

	objects[RED_DETECTOR] = detectors;
	objects[GREEN_DETECTOR] = detectors + 1;
	objects[BLUE_DETECTOR] = detectors + 2;
}

