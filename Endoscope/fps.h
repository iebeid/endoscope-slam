#include <cstdio>
#include <ctime>

class fps{
public:
	fps();
	~fps();
	void start_fps_counter();
	void end_fps_counter();
	void print_fps();

private:
	int frames;
	double starttime;
	bool first;
	float nof;
	std::clock_t start;
	double timepassed;
};