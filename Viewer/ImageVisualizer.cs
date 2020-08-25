using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Viewer
{
    class ImageVisualizer { 
        public System.Drawing.Bitmap Bitmap { get; set; }
        public System.Windows.Media.ImageSource ImageSource { get; set; }

        public ImageVisualizer(byte[] source, int width, int height) : this(GetBitmap(source, width, height)) { }
        public ImageVisualizer(System.Drawing.Bitmap bitmap)
        {
            this.Bitmap = bitmap;
            this.ImageSource = ImageSourceFromBitmap(bitmap);
        }
        public static unsafe ImageSource ImageSourceFromBitmap(Bitmap source)
        {
            if (source == null)
                return null;
            var pf = PixelFormats.Bgra32;
            switch (source.PixelFormat)
            {
                case System.Drawing.Imaging.PixelFormat.Format8bppIndexed:
                    pf = PixelFormats.Indexed8;
                    break;
                case System.Drawing.Imaging.PixelFormat.Format24bppRgb:
                    pf = PixelFormats.Rgb24;
                    break;                
                case System.Drawing.Imaging.PixelFormat.Format32bppArgb:
                    pf = PixelFormats.Bgra32;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
            BitmapPalette palette = null;
            if (source.PixelFormat == System.Drawing.Imaging.PixelFormat.Format8bppIndexed)
            {
                pf = PixelFormats.Indexed8;
                palette = new BitmapPalette(source.Palette.Entries.Select(x => System.Windows.Media.Color.FromArgb(x.A, x.R, x.G, x.B)).ToList());
            }
            var result = new WriteableBitmap(source.Width, source.Height, source.HorizontalResolution, source.VerticalResolution, pf, palette);
            var data = source.LockBits(new Rectangle(0, 0, source.Width, source.Height), ImageLockMode.ReadOnly, source.PixelFormat);
            var bytes = new byte[data.Height * data.Stride];
            Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);
            result.WritePixels(new Int32Rect(0, 0, source.Width, source.Height), bytes, data.Stride, 0);
            source.UnlockBits(data);

            return result;
        }
        static System.Drawing.Bitmap GetBitmap(byte[] source, int width, int height)
        {
            var bitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            var data = bitmap.LockBits(new Rectangle(0, 0, width, height), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            Marshal.Copy(source, 0, data.Scan0, source.Length);
            bitmap.UnlockBits(data);
            return bitmap;
        }
    }
}
