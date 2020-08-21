using System;
using System.Drawing;

namespace Viewer {
    public class Oilify {
        public void Execute(Bitmap bIn, out Bitmap bOut, int radius, float intensityLevels, int colorCount) {
            var lbIn = new LockBitmap(bIn);
            bOut = (Bitmap)bIn.Clone();
            var lbOut = new LockBitmap(bOut);
            lbIn.LockBits();
            lbOut.LockBits();
            ExecuteImpl(lbIn.Pixels, radius, intensityLevels, bIn.Width, bIn.Height, lbOut.Pixels, colorCount);
        }
        struct ColorRange {
            public int from;
            public int to;
            public int value;
        }
        void ExecuteImpl(byte[] bitmapIn,
                         int radius,
                         float intensityLevels,
                         int width,
                         int height,
                         byte[] bitmapOut, int colorsCount) {
            int[] nIntensityCount;
            int[] nSum;

            ColorRange[] colorRanges = new ColorRange[colorsCount];
            var colorRangeMax = 255;
            int colorRangeIndex = 0;
            var colorRangeDelta = (int) Math.Floor((colorRangeMax - 1) / (double) colorsCount);
            var colorRangeStart = 0;
            for (int i = 0; i < colorsCount; i ++) {

                var range = new ColorRange() {@from = colorRangeStart, to = colorRangeStart + colorRangeDelta * 2 / 3, value = colorRangeStart + colorRangeDelta / 3+3};
                colorRanges[colorRangeIndex] = range;
                colorRangeStart += colorRangeDelta;
            }

            // Border pixes( depends on nRadius) will become black.
            // On increasing radius boundary pixels should set as black.

            // If total bytes in a row of image is not divisible by four, 
            // blank bytes will be padded to the end of the row.
            // nBytesInARow bytes are the actual size of a row instead of nWidth * 3.
            // If width is 9, then actual bytes in a row will will be 28, and not 27.
            int nBytesInARow = width;

            // nRadius pixels are avoided from left, right top, and bottom edges.
            for (int nY = radius; nY < height - radius; nY++) {
                for (int nX = radius; nX < width - radius; nX++) {
                    // Reset calculations of last pixel.
                    nIntensityCount = new int[256];
                    nSum = new int[256];

                    // Find intensities of nearest nRadius pixels in four direction.
                    for (int nY_O = -radius; nY_O <= radius; nY_O++) {
                        for (int nX_O = -radius; nX_O <= radius; nX_O++) {
                            int n = colorRanges[bitmapIn[(nX + nX_O) + (nY + nY_O) * nBytesInARow]].value;

                            // Find intensity of RGB value and apply intensity level.
                            int nCurIntensity = (int) (n * intensityLevels / 255);
                            if (nCurIntensity > 255)
                                nCurIntensity = 255;
                            int i = nCurIntensity;
                            nIntensityCount[i]++;

                            nSum[i] = nSum[i] + n;
                        }
                    }

                    int nOut = 0;

                    int nCurMax = 0;
                    int nMaxIndex = 0;
                    for (int nI = 0; nI < 256; nI++) {
                        if (nIntensityCount[nI] > nCurMax) {
                            nCurMax = nIntensityCount[nI];
                            nMaxIndex = nI;
                        }
                    }

                    nOut = nSum[nMaxIndex] / nCurMax;

                    int nOutVal = 0;
                    for (int i = 0; i < colorRanges.Length; i++) {
                        var range = colorRanges[i];
                        if (nOut >= range.@from && nOut <= range.to) {
                            nOutVal = i;
                            break;
                        }
                    }

                    bitmapOut[(nX)  + (nY) * nBytesInARow] = (byte)nOutVal;
                }
            }
        }
    }
}