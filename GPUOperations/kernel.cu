
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void __global__ addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__device__ char GetClosestPaletteColorIndex(const char* palette, int r, int g, int b, char count) {
    float minDist = 100000;
    char index;
    for (char i = 0; i < count; i++) {
        float distance = sqrtf((powf(palette[i * 3 + 0] - r, 2) + powf(palette[i * 3 + 1] - g, 2) + powf(palette[i * 3 + 2] - b, 2)));
        if (distance < minDist) {
            minDist = distance;
            index = i;
        }
    }
    return index;
}

void __global__ calculatePixel(const char* input, char* output, const char* palette, const int width, const int height, const int radius, const float intensity, const char paletteLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nY = index / width;
    int nX = index % width;
    // Reset calculations of last pixel.
    float nIntensityCount[256];
    float nSumR[256];
    float nSumG[256];
    float nSumB[256];

    for (int i = 0; i < 256; i++) {
        nIntensityCount[i] = 0.0;
        nSumR[i] = 0.0;
        nSumG[i] = 0.0;
        nSumB[i] = 0.0;
    }
    

    // Find intensities of nearest nRadius pixels in four direction.
    for (int nY_O = -radius; nY_O <= radius; nY_O++) {
        for (int nX_O = -radius; nX_O <= radius; nX_O++) {
            int nY_S = nY + nY_O;
            int nX_S = nX + nX_O;
            int n = input[nX + nY * width];
            if (nY_S >= 0 && nY_S < height && nX_S >= 0 && nX_S < width) {
                n = input[nX_S + nY_S * width];
            }

            float nR = palette[n * 3 + 0];
            float nG = palette[n * 3 + 1];
            float nB = palette[n * 3 + 2];

            // Find intensity of RGB value and apply intensity level.
            float nCurIntensity = (int)((((float)(nR + nG + nB) / 3.0) * intensity) / 255.0);
            if (nCurIntensity > 255)
                nCurIntensity = 255;
            int i = nCurIntensity;
            nIntensityCount[i]++;

            nSumR[i] = nSumR[i] + nR;
            nSumG[i] = nSumG[i] + nG;
            nSumB[i] = nSumB[i] + nB;
        }
    }

    int nOutR = 0;
    int nOutG = 0;
    int nOutB = 0;

    int nCurMax = 0;
    int nMaxIndex = 0;
    for (int nI = 0; nI < 256; nI++) {
        if (nIntensityCount[nI] > nCurMax) {
            nCurMax = nIntensityCount[nI];
            nMaxIndex = nI;
        }
    }

    nOutR = (int)((float)nSumR[nMaxIndex] / (float)nCurMax);
    nOutG = (int)((float)nSumG[nMaxIndex] / (float)nCurMax);
    nOutB = (int)((float)nSumB[nMaxIndex] / (float)nCurMax);

    output[index] = GetClosestPaletteColorIndex(palette, nOutR, nOutG, nOutB, paletteLength);
}

extern "C" __declspec(dllexport) void __stdcall Oilify(int radius, float intensity, int width, int height, char* input, char* output, char* palette, char paletteLength) {
    char* devInput = nullptr;
    char* devOutput = nullptr;
    char* devPalette = nullptr;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);

    auto arraySize = width * height;

    cudaStatus = cudaMalloc(&devInput, sizeof(char) * arraySize);
    cudaStatus = cudaMalloc(&devOutput, sizeof(char) * arraySize);
    cudaStatus = cudaMalloc(&devPalette, sizeof(char) * paletteLength * 3);

    cudaStatus = cudaMemcpy(devInput, input, sizeof(char)*arraySize, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devPalette, palette, sizeof(char) * paletteLength * 3, cudaMemcpyHostToDevice);    
    int threadCount = 64;
    calculatePixel << <width*height/threadCount, threadCount>> > (devInput, devOutput, devPalette, width, height, radius, intensity, paletteLength);
    cudaStatus = cudaGetLastError();
    cudaStatus = cudaDeviceSynchronize();    
    cudaStatus = cudaMemcpy(output, devOutput, sizeof(char)*arraySize, cudaMemcpyDeviceToHost);

    cudaStatus = cudaFree(&devInput);
    cudaStatus = cudaFree(&devOutput);
    cudaStatus = cudaFree(&devPalette);
}


void __global__ TestDevice(char a, char b, char* c) {
#if __CUDA_ARCH__>=200
    printf("test\n");
#endif
    *c = a + b;
}

extern "C" __declspec(dllexport) void __stdcall Test(char a, char b, char* c) {
    char* cdev = nullptr;
    cudaMalloc(&cdev, sizeof(char));

    TestDevice << <1, 1 >> > (a, b, cdev);

    cudaDeviceSynchronize();

    cudaMemcpy(c, cdev, 1, cudaMemcpyDeviceToHost);
}