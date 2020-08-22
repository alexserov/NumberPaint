
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void __global__ addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/*
* byte[] input,
                          int radius,
                          float intensity,
                          int width,
                          int height,
                          byte[] output, Color[] palette, byte paletteLength
*/

__device__ char GetClosestPaletteColorIndex(char* palette, char r, char g, char b, char count) {
    float minDist = 1000;
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

void __global__ calculatePixel(char* input, char* output, char* palette, int width, int height, int radius, float intensity, char paletteLength) {
    int index = threadIdx.x;
    // Reset calculations of last pixel.
    int* nIntensityCount = (int*)malloc(sizeof(int) * 256);
    int* nSumR = (int*)malloc(sizeof(int) * 256);
    int* nSumG = (int*)malloc(sizeof(int) * 256);
    int* nSumB = (int*)malloc(sizeof(int) * 256);

    int nY = index / width;
    int nX = index % width;

    // Find intensities of nearest nRadius pixels in four direction.
    for (int nY_O = -radius; nY_O <= radius; nY_O++) {
        for (int nX_O = -radius; nX_O <= radius; nX_O++) {
            int nY_S = nY + nY_O;
            int nX_S = nX + nX_O;
            int n = input[nX + nY * width];
            if (nY_S >= 0 && nY_S < height && nX_S >= 0 && nX_S < width) {
                n = input[nX_S + nY_S * width];
            }

            int nR = palette[n];
            int nG = palette[n + 1];
            int nB = palette[n + 2];

            // Find intensity of RGB value and apply intensity level.
            int nCurIntensity = (int)((((nR + nG + nB) / 3.0) * intensity) / 255);
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

    nOutR = nSumR[nMaxIndex] / nCurMax;
    nOutG = nSumG[nMaxIndex] / nCurMax;
    nOutB = nSumB[nMaxIndex] / nCurMax;

    output[index] = GetClosestPaletteColorIndex(palette, nOutR, nOutG, nOutB, paletteLength);

    free(&nIntensityCount);
    free(&nSumR);
    free(&nSumG);
    free(&nSumB);
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

    cudaMemcpy(devInput, input, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(devPalette, palette, paletteLength * 3, cudaMemcpyHostToDevice);

    calculatePixel << <1, arraySize >> > (devInput, devOutput, devPalette, width, height, radius, intensity, paletteLength);

    cudaStatus = cudaDeviceSynchronize();

    cudaMemcpy(output, devOutput, arraySize, cudaMemcpyHostToDevice);

    cudaFree(&devInput);
    cudaFree(&devOutput);
    cudaFree(&devPalette);
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 0;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
