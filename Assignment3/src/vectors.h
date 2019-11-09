#pragma once

struct vector2d {
public:
	vector2d();
	vector2d(float X, float Y);
	vector2d& operator=(const vector2d& rhs);
	vector2d& operator+=(const vector2d& v);
	vector2d& operator-=(const vector2d& v);
	vector2d operator+(const vector2d& v);
	vector2d operator-(const vector2d& v);
	friend vector2d operator*(const float scalar, const vector2d& v);

	static vector2d normalize(const vector2d& v);
	static float magnitude(const vector2d& v);
	static float dot(const vector2d& u, const vector2d& v);

	float x;
	float y;
};

struct vector3d : public vector2d {
	vector3d() = default;
	vector3d(float X, float Y, float Z);
	vector3d& operator=(const vector3d& rhs);

	float z;
};