#pragma once
extern "C" __declspec(dllexport) void __stdcall Oilify(int radius, float intensity, int width, int height, char* input, char* output, char* palette, char paletteLength);
extern "C" __declspec(dllexport) void __stdcall BuildHeatMap(const unsigned char* data, unsigned long int* heatMap, int width, int height, unsigned long int* levels);