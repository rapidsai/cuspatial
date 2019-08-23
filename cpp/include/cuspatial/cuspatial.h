#pragma once

namespace cuspatial
{
	struct location_3d
	{
		double latitude;
		double lon;
		double alt;
	};

	struct Coord2D
	{
		double x;
		double y;
	};

	struct TimeStamp
	{
		uint32_t y : 6;
		uint32_t m : 4;
		uint32_t d : 5;
		uint32_t hh : 5;
		uint32_t mm : 6;
		uint32_t ss : 6;
		uint32_t wd: 3;
		uint32_t yd: 9;
		uint32_t ms: 10;
		uint32_t pid:10;
	};
}// namespace cuSpatial
