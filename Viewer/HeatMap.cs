using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;

namespace Viewer
{
    public class HeatMap
    {
        [DllImport("GPUOperations.dll")]
        static extern void BuildHeatMap(byte[] data, UInt32[] heatMap, int width, int height, ref UInt32 levels);

        public static (UInt32[] map,UInt32 levels) Build(Bitmap source)
        {
            var lb = new LockBitmap(source);
            try
            {                
                lb.LockBits();
                var result = new UInt32[lb.Pixels.Length];
                UInt32 levels = 0;
                BuildHeatMap(lb.Pixels, result, lb.Width, lb.Height, ref levels);
                return (result, levels);
            }
            finally
            {
                lb.UnlockBits();
            }
            

        }
    }
}
