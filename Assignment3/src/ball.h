#pragma once
#include "vectors.h"

struct ball {
public:
	ball() = default;
	ball(float r, vector2d c, vector2d vel, int m)
		:
		radius(r),
		center(c),
		velocity(vel),
		mass(m)
	{
		if (radius == 0.05f) color = vector3d(0.5f, 1.f, 0.5f);
		else if (radius == 0.1f) color = vector3d(0.5f, 0.5f, 1.f);
		else color = vector3d(1.f, 0.5f, 0.5f);
	}

	vector3d color;
	vector2d center;
	vector2d velocity;
	float radius;
	int mass;
};
