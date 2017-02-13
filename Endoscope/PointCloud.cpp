
#include "PointCloud.h"

#include <memory>
#include <vector>
#include <iostream>
#include <vector>

PointCloud::~PointCloud(){
	points.clear();
	points.shrink_to_fit();
}

PointCloud::PointCloud(cv::Mat * rgb_frame, cv::Mat * depth_frame, cv::Mat * mapped_rgb_frame, int depth_threshold, int point_cloud_resolution){

}

PointCloud::PointCloud(vector<Point> p){
	this->points = p;
	memcpy(&this->points[0], &p[0], p.size()*sizeof(Point));
}

Render PointCloud::get_rendering_structures(){
	Render rs;
	rs.vertices = new GLfloat[this->points.size() * 3];
	rs.colors = new GLfloat[this->points.size() * 3];
	rs.normal_colors = new GLfloat[this->points.size() * 3];
	int t = 0;
	Point p;
	for (int j = 0; j < this->points.size(); j++)
	{
		p = this->points[j];
		rs.vertices[t] = p.position.x;
		rs.vertices[t + 1] = p.position.y;
		rs.vertices[t + 2] = p.position.z;
		rs.colors[t] = p.color.blue;
		rs.colors[t + 1] = p.color.green;
		rs.colors[t + 2] = p.color.red;
		rs.normal_colors[t] = p.normal_color.blue;
		rs.normal_colors[t + 1] = p.normal_color.green;
		rs.normal_colors[t + 2] = p.normal_color.red;
		t = t + 3;
	}
	return rs;
}

PointCloud PointCloud::transform_glm(PointCloud mo, Transformation trans){
	PointCloud transformed;
	glm::vec3 translation = glm::vec3((float)trans.t.val[0][0], (float)trans.t.val[1][0], (float)trans.t.val[2][0]);
	glm::mat3 rot_mat = glm::mat3(trans.R.val[0][0], trans.R.val[0][1], trans.R.val[0][2], trans.R.val[1][0], trans.R.val[1][1], trans.R.val[1][2], trans.R.val[2][0], trans.R.val[2][1], trans.R.val[2][2]);
	Point p;
	PXCColor c;
	for (int i = 0; i < mo.points.size(); i++){
		glm::vec3 vertex(mo.points[i].position.x, mo.points[i].position.y, mo.points[i].position.z);
		glm::vec3 normal(mo.points[i].normal_vector.x, mo.points[i].normal_vector.y, mo.points[i].normal_vector.z);
		glm::vec3 new_vertex_vector = rot_mat * vertex + translation;
		glm::vec3 new_normal_vector = rot_mat * normal + translation;
		p.position.x = new_vertex_vector.x;
		p.position.y = new_vertex_vector.y;
		p.position.z = new_vertex_vector.z;
		p.normal_vector.x = new_normal_vector.x;
		p.normal_vector.y = new_normal_vector.y;
		p.normal_vector.z = new_normal_vector.z;
		c = mo.points[i].color;
		p.color.red = c.red;
		p.color.green = c.green;
		p.color.blue = c.blue;
		p.distance_from_origin = sqrt(pow(p.position.x, 2) + pow(p.position.y, 2) + pow(p.position.z, 2));
		transformed.points.push_back(p);
	}
	return transformed;
}

void PointCloud::transform(PointCloud mo, Transformation trans){
	Matrix translation(3, 1);
	translation.val[0][0] = (float)trans.t.val[0][0];
	translation.val[1][0] = (float)trans.t.val[0][1];
	translation.val[2][0] = (float)trans.t.val[0][2];
	Matrix rot_mat(3, 3);
	rot_mat.val[0][0] = trans.R.val[0][0];
	rot_mat.val[0][1] = trans.R.val[1][0];
	rot_mat.val[0][2] = trans.R.val[2][0];
	rot_mat.val[1][0] = trans.R.val[0][1];
	rot_mat.val[1][1] = trans.R.val[1][1];
	rot_mat.val[1][2] = trans.R.val[2][1];
	rot_mat.val[2][0] = trans.R.val[0][2];
	rot_mat.val[2][1] = trans.R.val[1][2];
	rot_mat.val[2][2] = trans.R.val[2][2];
	Point p;
	Matrix v(3, 1);
	Matrix n(3, 1);
	PXCColor c;
	Matrix new_vertex_vector;
	Matrix new_normal_vector;
	for (int i = 0; i < mo.points.size(); i++){
		v.val[0][0] = mo.points[i].position.x;
		v.val[1][0] = mo.points[i].position.y;
		v.val[2][0] = mo.points[i].position.z;
		n.val[0][0] = mo.points[i].normal_vector.x;
		n.val[1][0] = mo.points[i].normal_vector.y;
		n.val[2][0] = mo.points[i].normal_vector.z;
		c = mo.points[i].color;
		new_vertex_vector = rot_mat * v + translation;
		new_normal_vector = rot_mat * n + translation;
		p.position.x = (float)new_vertex_vector.val[0][0];
		p.position.y = (float)new_vertex_vector.val[1][0];
		p.position.z = (float)new_vertex_vector.val[2][0];
		p.normal_vector.x = (float)new_normal_vector.val[0][0];
		p.normal_vector.y = (float)new_normal_vector.val[1][0];
		p.normal_vector.z = (float)new_normal_vector.val[2][0];
		p.color.red = c.red;
		p.color.green = c.green;
		p.color.blue = c.blue;
		p.distance_from_origin = sqrt(pow(p.position.x, 2) + pow(p.position.y, 2) + pow(p.position.z, 2));
		this->points.push_back(p);
	}
}

void PointCloud::terminate(Render rs){
	free(rs.vertices);
	free(rs.colors);
	free(rs.normal_colors);
	this->~PointCloud();
}