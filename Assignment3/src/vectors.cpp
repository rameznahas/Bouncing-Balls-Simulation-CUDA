#include <math.h>
#include "vectors.h"

vector2d::vector2d() : x(0.0), y(0.0) {}
vector2d::vector2d(float X, float Y) : x(X), y(Y) {}

vector2d& vector2d::operator=(const vector2d& rhs) {
	if (&rhs != this) {
		x = rhs.x;
		y = rhs.y;
	}
	return *this;
}

vector2d& vector2d::operator+=(const vector2d& v) {
	x += v.x;
	y += v.y;
	return *this;
}

vector2d& vector2d::operator-=(const vector2d& v) {
	x -= v.x;
	y -= v.y;
	return *this;
}

vector2d vector2d::operator+(const vector2d& v) {
	return vector2d(x + v.x, y + v.y);
}

vector2d vector2d::operator-(const vector2d& v) {
	return vector2d(x - v.x, y - v.y);
}

vector2d operator*(const float scalar, const vector2d& v) {
	return vector2d(v.x * scalar, v.y * scalar);
}

vector2d vector2d::normalize(const vector2d& v) {
	float magnitude = sqrtf(powf(v.x, 2) + powf(v.y, 2));

	return vector2d(v.x / magnitude, v.y / magnitude);
}

float vector2d::magnitude(const vector2d& v) {
	return sqrtf(powf(v.x, 2) + powf(v.y, 2));
}

float vector2d::dot(const vector2d& u, const vector2d& v) {
	return u.x * v.x + u.y * v.y;
}

vector3d::vector3d(float X, float Y, float Z) : vector2d(X, Y), z(Z) {}

vector3d& vector3d::operator=(const vector3d& rhs) {
	if (&rhs != this) {
		vector2d::operator=(rhs);
		z = rhs.z;
	}
	return *this;
}