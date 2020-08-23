// GPUOperationsTestExecutor.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "..\GPUOperations\export.h"
int main()
{
    char input[] = { 0, 0, 2, 3, 0, 0, 3, 0, 2, 2, 0, 0, 1, 1, 4, 4, 1, 0, 0, 1, 1, 3, 1, 0, 3, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 6, 0, 0, 0, 0, 3, 3, 3, 0, 3, 4, 0, 0, 0, 1, 0, 0, 0, 0, 7, 6, 12, 11, 11, 2, 0, 1, 0, 0, 0, 4, 0, 0, 6, 10, 11, 11, 11, 11, 12, 12, 12, 3, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 8, 8, 11, 11, 11, 11, 9, 2, 0, 0, 0, 8, 8, 8, 11, 8, 11, 11, 8, 11, 8, 12, 8, 6, 0, 0, 0, 11, 11, 11, 8, 13, 11, 12, 12, 8, 11, 12, 11, 3, 0, 0, 0, 0, 6, 8, 6, 8, 8, 8, 2, 8, 12, 11, 0, 0, 0, 0, 0, 0, 8, 6, 6, 8, 12, 8, 5, 11, 12, 11, 3, 0, 0, 0, 0, 0, 0, 2, 6, 5, 5, 6, 5, 8, 6, 11, 6, 0, 0, 0, 0, 0, 0, 2, 6, 6, 6, 2, 5, 6, 6, 5, 3, 0, 6, 0, 2, 2, 0, 0, 0, 0, 6, 2, 2, 5, 6, 1, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 6, 2, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 6 };
    char output[sizeof(input)];
    char radius = 1;
    char intensity = 20;
    char width = 16;
    char height = 16;
    char palette[] = { 28, 35, 19, 50, 70, 38, 88, 46, 21, 87, 71, 39, 91, 97, 82, 146, 47, 7, 156, 82, 24, 156, 119, 69, 214, 102, 19, 255, 111, 65, 190, 130, 67, 248, 152, 33, 244, 163, 80, 249, 203, 92, 0, 255, 255, 255, 255, 255 };
    Oilify(radius, intensity, width, height, input, output, palette, 16);
    
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
