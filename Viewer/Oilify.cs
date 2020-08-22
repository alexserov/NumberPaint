using System;
using System.Drawing;
using System.Numerics;

namespace Viewer {
    public class Oilify {
        public void Execute(Bitmap bIn, out Bitmap bOut, int radius, float intensityLevels, int colorCount) {
            var lbIn = new LockBitmap(bIn);
            bOut = (Bitmap)bIn.Clone();
            var lbOut = new LockBitmap(bOut);
            lbIn.LockBits();
            lbOut.LockBits();
            ExecuteImpl2(lbIn.Pixels, radius, intensityLevels, bIn.Width, bIn.Height, lbOut.Pixels, bIn.Palette.Entries, (byte)colorCount);
            lbIn.UnlockBits();
            lbOut.UnlockBits();
        }
        struct ColorRange {
            public int from;
            public int to;
            public int value;
        }

        void ExecuteImpl2(byte[] pbyDataIn_i,
                          int nRadius_i,
                          float fIntensityLevels_i,
                          int nWidth_i,
                          int nHeight_i,
                          byte[] pbyDataOut_o, Color[] palette, byte paletteLength) {
            var nIntensityCount = new int[256];
            var nSumR = new int[256];
            var nSumG = new int[256];
            var nSumB = new int[256];

            var nBytesInARow = nWidth_i;


            // nRadius pixels are avoided from left, right top, and bottom edges.
            for (int nY = nRadius_i; nY < nHeight_i - nRadius_i; nY++) {
                for (int nX = nRadius_i; nX < nWidth_i - nRadius_i; nX++) {
                    // Reset calculations of last pixel.
                    nIntensityCount = new int[256];
                    nSumR = new int[256];
                    nSumG = new int[256];
                    nSumB = new int[256];

                    // Find intensities of nearest nRadius pixels in four direction.
                    for (int nY_O = -nRadius_i; nY_O <= nRadius_i; nY_O++) {
                        for (int nX_O = -nRadius_i; nX_O <= nRadius_i; nX_O++) {
                            var nY_S = nY + nY_O;
                            var nX_S = nX + nX_O;
                            var n = pbyDataIn_i[nX+nY*nBytesInARow];
                            if (nY_S >= 0 && nY_S < nHeight_i && nX_S >= 0 && nX_S < nWidth_i) {
                                n = pbyDataIn_i[nX_S + nY_S * nBytesInARow];
                            }
                            var nC = palette[n];   
                            int nR = nC.R;
                            int nG = nC.G;
                            int nB = nC.B;

                            // Find intensity of RGB value and apply intensity level.
                            int nCurIntensity = (int) ((((nR + nG + nB) / 3.0) * fIntensityLevels_i) / 255);
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

                    pbyDataOut_o[(nX) + (nY) * nBytesInARow] = GetClosestPaletteColorIndex(palette, Color.FromArgb(255, nOutR, nOutG, nOutB), paletteLength);
                }
            }

            byte GetClosestPaletteColorIndex(Color[] palette, Color target, byte count) {
                var results = new double[count];
                for (int i = 0; i < count; i++) {
                    var currentColor = palette[i];
                    results[i] = Math.Sqrt(Math.Pow(currentColor.R - target.R, 2) + Math.Pow(currentColor.G - target.G, 2) + Math.Pow(currentColor.B - target.B, 2));
                }

                var min = double.MaxValue;
                byte minIndex = 0;
                for (byte i = 0; i < count; i++) {
                    if (results[i] < min) {
                        min = results[i];
                        minIndex = i;
                    }
                }

                return minIndex;
            }

            void ExecuteImpl(byte[] bitmapIn,
                             int radius,
                             float intensityLevels,
                             int width,
                             int height,
                             byte[] bitmapOut, int colorsCount) {

                var surround = new byte[(radius * 2 + 1) * (radius * 2 + 1)];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        var index = y * width + x;
                        var currentColor = bitmapIn[index];
                        for (int iR = -radius; iR <= radius; iR++) {
                            for (int jR = -radius; jR <= radius; jR++) {
                                var column = y + iR;
                                var row = x + jR;

                                var indexSurround = (iR + radius) * (radius * 2 + 1) + jR + radius;
                                if (row < 0 || row >= height || column < 0 || column >= width) {
                                    surround[indexSurround] = currentColor;
                                    continue;
                                }

                                var indexR = row * width + row;

                                surround[indexSurround] = bitmapIn[indexR];
                            }
                        }

                        var popular = findPopular(surround);
                        bitmapOut[index] = popular;
                    }
                }
            }

             byte findPopular(byte[] a) {

                if (a == null || a.Length == 0)
                    return 0;

                Array.Sort(a);

                byte previous = a[0];
                byte popular = a[0];
                int count = 1;
                int maxCount = 1;

                for (int i = 1; i < a.Length; i++) {
                    if (a[i] == previous)
                        count++;
                    else {
                        if (count > maxCount) {
                            popular = a[i - 1];
                            maxCount = count;
                        }

                        previous = a[i];
                        count = 1;
                    }
                }

                return count > maxCount ? a[a.Length - 1] : popular;

            }
        }

    }
}