
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_60_atomic_functions.h"

#include <stdio.h>

//#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//__device__ double atomicAdd(double* a, double b) { return b; }
//__device__ unsigned long atomicAdd(unsigned long* a, unsigned long b) { return b; }
//__device__ unsigned long long atomicAdd(unsigned long long* a, unsigned long long b) { return b; }
//#else
//#endif

__device__ unsigned char GetClosestPaletteColorIndex(const unsigned char* palette, int r, int g, int b, unsigned char count) {
    float minDist = 100000;
    unsigned char index;
    for (unsigned char i = 0; i < count; i++) {
        float distance = sqrtf((powf(palette[i * 3 + 0] - r, 2) + powf(palette[i * 3 + 1] - g, 2) + powf(palette[i * 3 + 2] - b, 2)));
        if (distance < minDist) {
            minDist = distance;
            index = i;
        }
    }
    return index;
}

void __global__ calculatePixel(const unsigned char* input, unsigned char* output, const unsigned char* palette, const int width, const int height, const int radius, const float intensity, const unsigned char paletteLength) {
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
            if (sqrtf(nY_O * nY_O + nX_O * nX_O) > radius)
                continue;
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
            float nCurIntensity = ((((float)(nR + nG + nB) / 3.0) * intensity) / 255.0);
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

extern "C" __declspec(dllexport) void __stdcall Oilify(int radius, float intensity, int width, int height,unsigned char* input, unsigned char* output, unsigned char* palette, unsigned char paletteLength) {
    unsigned char* devInput = nullptr;
    unsigned char* devOutput = nullptr;
    unsigned char* devPalette = nullptr;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);

    auto arraySize = width * height;

    cudaStatus = cudaMalloc(&devInput, sizeof(unsigned char) * arraySize);
    cudaStatus = cudaMalloc(&devOutput, sizeof(unsigned char) * arraySize);
    cudaStatus = cudaMalloc(&devPalette, sizeof(unsigned char) * paletteLength * 3);

    cudaStatus = cudaMemcpy(devInput, input, sizeof(unsigned char)*arraySize, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devPalette, palette, sizeof(unsigned char) * paletteLength * 3, cudaMemcpyHostToDevice);    
    int threadCount = 16;
    calculatePixel << <width*height/threadCount, threadCount>> > (devInput, devOutput, devPalette, width, height, radius, intensity, paletteLength);
    cudaStatus = cudaGetLastError();
    cudaStatus = cudaDeviceSynchronize();     
    cudaStatus = cudaMemcpy(output, devOutput, sizeof(unsigned char)*arraySize, cudaMemcpyDeviceToHost);

    cudaStatus = cudaFree(&devInput);
    cudaStatus = cudaFree(&devOutput);
    cudaStatus = cudaFree(&devPalette);
}


__device__ unsigned long int processHeatMap_iteration;
__device__ unsigned long long int processHeatMap_count;
__device__ unsigned int processHeatMap_width;
__device__ unsigned int processHeatMap_height;
void __global__ processHeatMap(const unsigned char* data, unsigned long int* inputMap, unsigned long int* outputMap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int width = processHeatMap_width;
    unsigned int height = processHeatMap_height;
    int y = index / width;
    int x = index % width;

    unsigned char dataValue = data[index];
    unsigned long int mapValue = inputMap[index];
    bool success = true;
    for (int i = -1; i <= 1; i++) {
        if (!success)
            break;
        for (int j = -1; j <= 1; j++) {
            int yN = y + i;
            int xN = x + j;
            int indexN = xN + yN * width;
            if (yN < 0 || xN < 0 || yN >= height || xN >= width) {
                success = false;
                break;
            }                
            if (data[indexN] != dataValue || inputMap[indexN]!=processHeatMap_iteration) {
                success = false;
                break;
            }
        }
    }
    if (success) {
        outputMap[index] = mapValue + 1;
        atomicAdd(&processHeatMap_count, 1);
    }    
}

extern "C" __declspec(dllexport) void __stdcall BuildHeatMap(const unsigned char* data, unsigned long int* heatMap, unsigned int width, unsigned int height, unsigned long int* levels) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);

    unsigned char* dataDev = nullptr;
    unsigned  long int* inputHeatMapDev = nullptr;
    unsigned  long int* outputHeatMapDev = nullptr;
    unsigned long long int countHost;
    int size = width * height;
    unsigned long int iteration = 0;

    //symbols    
    unsigned long int* iterationDev = nullptr;
    unsigned long long int* countDev = nullptr;
    unsigned int* widthDev = nullptr;
    unsigned int* heightDev = nullptr;

    cudaStatus = cudaGetSymbolAddress((void**)&iterationDev, processHeatMap_iteration);
    cudaStatus = cudaGetSymbolAddress((void**)&countDev, processHeatMap_count);
    cudaStatus = cudaGetSymbolAddress((void**)&widthDev, processHeatMap_width);
    cudaStatus = cudaGetSymbolAddress((void**)&heightDev, processHeatMap_height);

    //arrays
    memset(heatMap, 0, size * sizeof(unsigned long int));

    cudaStatus = cudaMalloc(&dataDev, size);
    cudaStatus = cudaMalloc(&inputHeatMapDev, size * sizeof(unsigned long int));
    cudaStatus = cudaMalloc(&outputHeatMapDev, size * sizeof(unsigned long int));
    
    
    cudaStatus = cudaMemcpy(dataDev, data, size, cudaMemcpyHostToDevice);    
    cudaStatus = cudaMemcpy(outputHeatMapDev, heatMap, size * sizeof(unsigned long int), cudaMemcpyHostToDevice);
    
    //values        
    cudaStatus = cudaMemcpy(widthDev, &width, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(heightDev, &height, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemset(iterationDev, 0, sizeof(unsigned long int));
    
    do {
        countHost = 0;
        cudaStatus = cudaMemcpy(iterationDev, &iteration, sizeof(unsigned long int), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(inputHeatMapDev, outputHeatMapDev, size * sizeof(unsigned long int), cudaMemcpyDeviceToDevice);
        cudaStatus = cudaMemset(countDev, 0, sizeof(unsigned long int));
        int threadCount = 16;
        processHeatMap << <size / threadCount, threadCount >> > (dataDev, inputHeatMapDev, outputHeatMapDev);
        cudaStatus = cudaGetLastError();
        cudaStatus = cudaDeviceSynchronize();
        cudaMemcpy(&countHost, countDev, sizeof(unsigned long int), cudaMemcpyDeviceToHost);
        iteration++;
    } while (countHost != 0);
    cudaMemcpy(heatMap, outputHeatMapDev, size*sizeof(unsigned long int), cudaMemcpyDeviceToHost);
    *levels = iteration - 1;
}