#pragma once

typedef unsigned int uint;


typedef struct Location
{
    double lat;
    double lon;
    double alt;
} Location;

typedef struct Coord
{
    double x;
    double y;
} Coord;

typedef struct Time
{
    uint y : 6;
    uint m : 4;
    uint d : 5;
    uint hh : 5;
    uint mm : 6;
    uint ss : 6;
    uint wd: 3;
    uint yd: 9;
    uint ms: 10;
    uint pid:10;
}Time;