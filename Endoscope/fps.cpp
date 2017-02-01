#include "fps.h"
#include <iostream>

using namespace std;

fps::fps(){
	frames = 0;
	starttime = 0;
	timepassed = 1;
	first = true;
	nof = 0.0f;
	start = std::clock();
}

fps::~fps(){

}

void fps::start_fps_counter(){
	if (first)
	{
		frames = 0;
		starttime = timepassed;
		first = false;
	}
	frames++;
}

void fps::end_fps_counter(){
	timepassed = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	if (timepassed - starttime > 0.25 && frames > 10)
	{
		nof = (float)frames / ((float)timepassed - (float)starttime);
		starttime = timepassed;
		frames = 0;
	}
}

void fps::print_fps(){
	cout << "Frame Rate: " << (int)nof;
	cout.flush();
	cout << '\r';
}
