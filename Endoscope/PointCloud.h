#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H 1

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#undef APIENTRY

#include "Matrix.h"

using namespace std;

struct PXCColor{
	float blue;
	float green;
	float red;
};

struct Point
{
	vector3f position;
	vector3f normal_vector;
	PXCColor color;
	PXCColor normal_color;
	float distance_from_origin;
};

struct Render{
	GLfloat * vertices;
	GLfloat * colors;
	GLfloat * normal_colors;
};

class Transformation{
public:
	Transformation(){
		R = Matrix::eye(3);
		t = Matrix(3, 1);
	}
public:
	Matrix R;
	Matrix t;
};

class PointCloud
{
public:

	PointCloud(){ ; }
	PointCloud(vector<Point> p);
	PointCloud(cv::Mat * rgb_frame, cv::Mat * depth_frame, cv::Mat * mapped_rgb_frame, int depth_threshold, int point_cloud_resolution);
	~PointCloud();

	void transform(PointCloud mo, Transformation trans);
	static PointCloud transform_glm(PointCloud mo, Transformation trans);
	Render get_rendering_structures();
	void terminate(Render rs);

public:

	std::vector<Point> points;
};

struct GlobalMap{
	vector<PointCloud> point_clouds;
};
#endif