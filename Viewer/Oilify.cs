using System;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Windows.Input;

namespace Viewer
{
    public class Oilify
    {
        public void Execute(Bitmap bIn, out Bitmap bOut, int radius, float intensityLevels, int colorCount, bool cuda)
        {
            var lbIn = new LockBitmap(bIn);
            bOut = (Bitmap)bIn.Clone();
            var lbOut = new LockBitmap(bOut);
            lbIn.LockBits();
            lbOut.LockBits();
            if (cuda)
                ExecuteImplUnmanaged(lbIn.Pixels, radius, intensityLevels, bIn.Width, bIn.Height, lbOut.Pixels, bIn.Palette.Entries, (byte)colorCount);
            else
                ExecuteImpl(lbIn.Pixels, radius, intensityLevels, bIn.Width, bIn.Height, lbOut.Pixels, bIn.Palette.Entries, (byte)colorCount);
            lbIn.UnlockBits();
            lbOut.UnlockBits();
        }

        struct ColorRange
        {
            public int from;
            public int to;
            public int value;
        }

        void ExecuteImpl(byte[] input,
                          int radius,
                          float intensity,
                          int width,
                          int height,
                          byte[] output, Color[] paletteIn, byte paletteLength)
        {
            var nIntensityCount = new int[256];
            var nSumR = new float[256];
            var nSumG = new float[256];
            var nSumB = new float[256];

            var nBytesInARow = width;

            byte[] palette = new byte[paletteLength * 3];
            for (int i = 0; i < paletteLength; i++)
            {
                palette[i * 3 + 0] = (byte)paletteIn[i].R;
                palette[i * 3 + 1] = (byte)paletteIn[i].G;
                palette[i * 3 + 2] = (byte)paletteIn[i].B;
            }


            // nRadius pixels are avoided from left, right top, and bottom edges.
            for (int nY = radius; nY < height - radius; nY++)
            {
                for (int nX = radius; nX < width - radius; nX++)
                {
                    // Reset calculations of last pixel.
                    nIntensityCount = new int[256];
                    nSumR = new float[256];
                    nSumG = new float[256];
                    nSumB = new float[256];             

                    // Find intensities of nearest nRadius pixels in four direction.
                    for (int nY_O = -radius; nY_O <= radius; nY_O++)
                    {
                        for (int nX_O = -radius; nX_O <= radius; nX_O++)
                        {
                            int nY_S = nY + nY_O;
                            int nX_S = nX + nX_O;
                            int n = input[nX + nY * width];
                            if (nY_S >= 0 && nY_S < height && nX_S >= 0 && nX_S < width)
                            {
                                n = input[nX_S + nY_S * width];
                            }

                            float nR = palette[n * 3 + 0];
                            float nG = palette[n * 3 + 1];
                            float nB = palette[n * 3 + 2];

                            // Find intensity of RGB value and apply intensity level.
                            float nCurIntensity = (int)((((float)(nR + nG + nB) / 3.0) * intensity) / 255.0);
                            if (nCurIntensity > 255)
                                nCurIntensity = 255;
                            int i = (int)nCurIntensity;
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
                    for (int nI = 0; nI < 256; nI++)
                    {
                        if (nIntensityCount[nI] > nCurMax)
                        {
                            nCurMax = nIntensityCount[nI];
                            nMaxIndex = nI;
                        }
                    }

                    nOutR = (int)((float)nSumR[nMaxIndex] / (float)nCurMax);
                    nOutG = (int)((float)nSumG[nMaxIndex] / (float)nCurMax);
                    nOutB = (int)((float)nSumB[nMaxIndex] / (float)nCurMax);

                    output[nX + nY * width] = GetClosestPaletteColorIndex(palette, nOutR, nOutG, nOutB, paletteLength);
                }
            }

            byte GetClosestPaletteColorIndex(byte[] palette, int r, int g, int b, byte count)
            {
                double minDist = 1000;
                byte minIndex = 0;
                for (byte i = 0; i < count; i++)
                {
                    var distance = Math.Sqrt(Math.Pow(palette[i*3+0] - r, 2) + Math.Pow(palette[i*3+1] - g, 2) + Math.Pow(palette[i*3+2] - b, 2));
                    if (distance < minDist)
                    {
                        minDist = distance;
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
                             byte[] bitmapOut, int colorsCount)
            {

                var surround = new byte[(radius * 2 + 1) * (radius * 2 + 1)];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        var index = y * width + x;
                        var currentColor = bitmapIn[index];
                        for (int iR = -radius; iR <= radius; iR++)
                        {
                            for (int jR = -radius; jR <= radius; jR++)
                            {
                                var column = y + iR;
                                var row = x + jR;

                                var indexSurround = (iR + radius) * (radius * 2 + 1) + jR + radius;
                                if (row < 0 || row >= height || column < 0 || column >= width)
                                {
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

            byte findPopular(byte[] a)
            {

                if (a == null || a.Length == 0)
                    return 0;

                Array.Sort(a);

                byte previous = a[0];
                byte popular = a[0];
                int count = 1;
                int maxCount = 1;

                for (int i = 1; i < a.Length; i++)
                {
                    if (a[i] == previous)
                        count++;
                    else
                    {
                        if (count > maxCount)
                        {
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

        [DllImport("GPUOperations.dll", EntryPoint = "Oilify")]
        static extern void OilifyImpl(int radius, float intensity, int width, int height, byte[] input, byte[] output, byte[] palette, byte paletteLength);

        [DllImport("GPUOperations.dll")]
        public static extern void Test(int x, int y, ref int result);

        void ExecuteImplUnmanaged(byte[] input,
                          int radius,
                          float intensity,
                          int width,
                          int height,
                          byte[] output, Color[] palette, byte paletteLength)
        {
            byte[] charPalette = new byte[paletteLength * 3];
            for (int i = 0; i < paletteLength; i++)
            {
                charPalette[i * 3 + 0] = (byte)palette[i].R;
                charPalette[i * 3 + 1] = (byte)palette[i].G;
                charPalette[i * 3 + 2] = (byte)palette[i].B;
            }
            var sb = new System.Text.StringBuilder();
            for(int i = 0; i< input.Length; i++)
            {
                sb.Append(input[i]);
                sb.Append(", ");
            }
            sb.AppendLine();
            sb.Append(radius);
            sb.AppendLine();
            sb.Append(intensity);
            sb.AppendLine();
            sb.Append(width);
            sb.AppendLine();
            sb.Append(height);
            for (int i = 0; i < charPalette.Length; i++)
            {
                sb.Append(charPalette[i]);
                sb.Append(", ");
            }
            Debug.WriteLine(sb.ToString());
            OilifyImpl(radius, intensity, width, height, input, output, charPalette, paletteLength);
        }

        byte GetClosestPaletteColorIndex(byte[] palette, byte r, byte g, byte b, byte count)
        {
            double minDist = 100000;
            byte index = 0;
            for (byte i = 0; i < count; i++)
            {
                double distance = Math.Sqrt((Math.Pow(palette[i * 3 + 0] - r, 2) + Math.Pow(palette[i * 3 + 1] - g, 2) + Math.Pow(palette[i * 3 + 2] - b, 2)));
                if (distance < minDist)
                {
                    minDist = distance;
                    index = i;
                }
            }
            return index;
        }
    }

}