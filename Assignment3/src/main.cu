#include <time.h>
#include <glew.h>
#include <freeglut.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <string>
#include <random>
#include <math.h>
#include "ball.h"

#define WWIDTH 800
#define WHEIGHT 800
#define UPDATE_FREQ 1.f / 30
#define BALL_COUNT 10
#define MIN_RADIUS 0.05f
#define PI 3.141592f
#define DEGREE_TO_RAD PI / 180
#define NUM_POINTS 360
#define BLOCK_SIZE 256

ball* balls, *d_balls;
ball** d_pairs;
size_t balls_count, pairs_count;
size_t balls_size, pairs_size;
size_t balls_grid_size, pairs_grid_size = 0;

vector2d GRAVITY(0.f, -1.5f);

clock_t previous_t = 0, current_t = 0;
float delta_t = UPDATE_FREQ;

__global__ void init_pairs(ball** d_pairs, ball* d_balls, size_t balls_count) {
	int count = 0;
	for (unsigned int i = 0; i < balls_count; ++i) {
		for (unsigned int j = (i + 1); j < balls_count; ++j) {
			d_pairs[count++] = &d_balls[i];
			d_pairs[count++] = &d_balls[j];
		}
	}
}

void init(int argc, char **argv) {
	//////////////////////////init display//////////////////////////
	glutInit(&argc, argv);
	glutInitWindowPosition(-1, -1);
	glutInitWindowSize(WWIDTH, WHEIGHT);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
	glutCreateWindow("Bouncing Balls Simulation");
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	////////////////////////////////////////////////////////////////

	///////////////////////////init balls///////////////////////////
	balls_count = argc > 1 ? std::stoi(argv[1]) : BALL_COUNT;
	balls = new ball[balls_count];

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> rad(1, 3);
	std::uniform_real_distribution<float> vel(-1.f, 1.f);

	for (unsigned int i = 0; i < balls_count; ++i) {
		int test = rad(gen);
		float radius = MIN_RADIUS * test; // random radius

		float ur_bound = radius - 1;
		float ll_bound = 1 - radius;

		std::uniform_real_distribution<float> coord(ur_bound, ll_bound); // so we dont get balls out of bounds
		vector2d center(coord(gen), coord(gen));

		int weight = (int)(radius * 100.0f);
		vector2d velocity(vel(gen), vel(gen));
		balls[i] = ball(radius, center, velocity, weight);
	}

	balls_size = balls_count * sizeof(ball);
	cudaMalloc(&d_balls, balls_size);
	cudaMemcpy(d_balls, balls, balls_size, cudaMemcpyHostToDevice);
	balls_grid_size = (size_t)std::ceilf((float)balls_count / BLOCK_SIZE);
	////////////////////////////////////////////////////////////////

	///////////////////////////init pairs///////////////////////////
	for (unsigned int i = 0; i < balls_count; ++i) {
		for (unsigned int j = (i + 1); j < balls_count; ++j) {
			++pairs_count;
		}
	}

	cudaMalloc(&d_pairs, pairs_count * sizeof(ball*) * 2);
	init_pairs<<<1, 1>>>(d_pairs, d_balls, balls_count);
	pairs_grid_size = (size_t)std::ceilf((float)pairs_count / BLOCK_SIZE);
	////////////////////////////////////////////////////////////////
}

__global__ void wall_bounce(ball* d_balls, float delta_t, size_t balls_count, vector2d GRAVITY) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < balls_count) {
		ball& current = d_balls[id];

		current.velocity.y += delta_t * GRAVITY.y;
		current.center.x += delta_t * current.velocity.x;
		current.center.y += delta_t * current.velocity.y;

		float t_wall = 1.f - current.radius;
		float b_wall = current.radius - 1.f;
		float r_wall = t_wall;
		float l_wall = b_wall;

		if (current.center.x > r_wall) {
			current.center.x = r_wall;
			current.velocity.x *= -1.f;
		}
		else if (current.center.x < l_wall) {
			current.center.x = l_wall;
			current.velocity.x *= -1.f;
		}

		if (current.center.y > t_wall) {
			current.center.y = t_wall;
			current.velocity.y *= -1.f;
		}
		else if (current.center.y < b_wall) {
			current.center.y = b_wall;
			current.velocity.y *= -1.f;
		}
	}
}

__global__ void ball_bounce(ball** d_pairs, size_t pairs_count) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < pairs_count) {
		unsigned int stride = 2 * id;
		ball* current = d_pairs[stride];
		ball* other = d_pairs[stride + 1];

		float min_dist = current->radius + other->radius;

		// check for aabb overlap
		// if true, balls are close enough, computation is worth it.
		if (current->center.x + min_dist > other->center.x
			&& current->center.y + min_dist > other->center.y
			&& other->center.x + min_dist > current->center.x
			&& other->center.y + min_dist > current->center.y) {

			float c_x = current->center.x - other->center.x;
			float c_y = current->center.y - other->center.y;
			float c = powf(c_x, 2.f) + powf(c_y, 2.f);

			// balls are close enough, but it does not mean they have collided.
			// check for ball collision.
			// if true, collision occured, handle it
			if (c <= powf(min_dist, 2.f)) {
				float distance = sqrtf(c);
				float overlap = 0.5f * (distance - current->radius - other->radius);

				float dir_x = c_x / distance;
				float dir_y = c_y / distance;

				current->center.x -= overlap * dir_x;
				current->center.y -= overlap * dir_y;
				other->center.x += overlap * dir_x;
				other->center.y += overlap * dir_y;

				float v_x = current->velocity.x - other->velocity.x;
				float v_y = current->velocity.y - other->velocity.y;
				int m = current->mass + other->mass;
				float mag = powf(distance, 2.f);
				float dot_vc = v_x * c_x + v_y * c_y;
				float ratio = 2.f * dot_vc / (m * mag);

				current->velocity.x -= (other->mass * ratio * c_x);
				current->velocity.y -= (other->mass * ratio * c_y);
				other->velocity.x += (current->mass * ratio * c_x);
				other->velocity.y += (current->mass * ratio * c_y);
			}
		}
	}
}

void draw() {
	glClearColor(0.25f, 0.25f, 0.25f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < balls_count; ++i) {
		ball& ball = balls[i];

		glBegin(GL_POLYGON);
		glColor4f(ball.color.x, ball.color.y, ball.color.z, 0.25f);

		for (int i = 0; i < NUM_POINTS; ++i) {
			float angle = i * DEGREE_TO_RAD;

			glVertex2d
			(
				ball.radius * cos(angle) + ball.center.x,	// x-coord
				ball.radius * sin(angle) + ball.center.y	// y-coord
			);
		}

		glEnd();
	}
	glutSwapBuffers();
}

void update() {
	//update current clock time
	current_t = clock();
	delta_t = (float)(current_t - previous_t) / CLOCKS_PER_SEC;

	// don't draw if delta_t is faster than 30 fps
	if (delta_t < UPDATE_FREQ) return;

	// store last draw time
	previous_t = current_t;

	wall_bounce<<<balls_grid_size, BLOCK_SIZE>>>(d_balls, delta_t, balls_count, GRAVITY);
	ball_bounce<<<pairs_grid_size, BLOCK_SIZE >>>(d_pairs, pairs_count);
	cudaMemcpy(balls, d_balls, balls_size, cudaMemcpyDeviceToHost);

	draw();
}

int main(int argc, char **argv) {
	init(argc, argv);
	glutDisplayFunc(update);
	glutIdleFunc(update);
	glutMainLoop();

	delete[] balls;
	cudaFree(d_balls);
	cudaFree(d_pairs);

	return 0;
}