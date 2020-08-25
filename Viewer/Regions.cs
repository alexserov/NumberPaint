using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Microsoft.SqlServer.Server;

namespace Viewer {
    internal class RegionWorker {
        public static Bitmap GetRegions(Bitmap source, int small)
        {
            source = (Bitmap)source.Clone();
            var reduced = (Bitmap)source.Clone();
            var lb = new LockBitmap(source);
            lb.LockBits();
            var bi = new BI { H = (ushort)source.Height, pixels = new PI[source.Width, source.Height], W = (ushort)source.Width };
            int totalColors = source.Palette.Entries.Length;
            for (int i = 0; i < source.Width; i++)
            {
                for (int j = 0; j < source.Height; j++)
                {
                    var color = lb.GetPixel(i, j);
                    bi.pixels[i, j] = new PI(i, j, color);
                }
            }
            lb.UnlockBits();
            lb = new LockBitmap(reduced);
            lb.LockBits();

            Console.WriteLine($"\n==========");
            Console.WriteLine($"{totalColors} colors found");
            bi.segmentMap = new int[source.Width, source.Height];
            List<SI> segments = new List<SI>();
            for (int i = 0; i < source.Width; i++)
            {
                for (int j = 0; j < source.Height; j++)
                {
                    var segmentIndex = bi.segmentMap[i, j];
                    if (segmentIndex == default)
                    {
                        segmentIndex = segments.Count + 1;
                        Stack<PI> st = new Stack<PI>();
                        List<PI> segmentData = new List<PI>();
                        st.Push(bi.pixels[i, j]);
                        while (st.Count != 0)
                        {
                            var curr = st.Pop();
                            bi.segmentMap[curr.X, curr.Y] = segmentIndex;
                            segmentData.Add(curr);
                            var ok = false;
                            for (var di = -1; di <= 1; di++)
                            {
                                for (var dj = -1; dj <= 1; dj++)
                                {
                                    if (di == 0 && dj == 0)
                                        continue;
                                    var next = curr.Move(di, dj, bi, out ok);
                                    if (ok && next.colorIndex == curr.colorIndex && bi.segmentMap[next.X, next.Y] == default)
                                    {
                                        st.Push(next);
                                    }
                                }
                            }
                        }

                        segments.Add(new SI(segmentData.ToArray(), segmentIndex, segmentData[0].colorIndex));
                        Console.Write($"\r{segmentIndex} segments found");
                    }
                }
            }
            Console.WriteLine($"\n==========");

            var smallTotal = segments.Count(x => x.items.Length < small);
            Console.WriteLine($"{smallTotal} small segments total");
            int currentSmall = 1;
            var segmentFromIndex = segments.ToDictionary(x => x.index);
            while (true)
            {
                var smallCount = segments.Count(x => x.items.Length < small);
                if (smallCount == 0)
                    break;
                Console.Write($"\rprocessing segments, {smallCount} left");
                currentSmall++;
                var orderedSegments = new Queue<SI>(segments.OrderBy(x => x.items.Length));
                var curr = orderedSegments.Peek();
                if (curr.items.Length >= small)
                    break;
                orderedSegments.Dequeue();
                var currFirstPoint = curr.items[0];
                var currIndex = curr.index;
                int shortpath = Int32.MaxValue;
                var targetSegment = curr;
                for (var di = -1; di <= 1; di++)
                {
                    for (var dj = -1; dj <= 1; dj++)
                    {
                        if (di == 0 && dj == 0)
                            continue;
                        if (shortpath == 1)
                            break;
                        var next = currFirstPoint.Move(di, dj, bi, out var success);
                        var path = 0;
                        while (success)
                        {
                            if (bi.segmentMap[next.X, next.Y] != currIndex)
                            {
                                break;
                            }

                            next = next.Move(di, dj, bi, out success);
                            path++;
                        }

                        if (success)
                        {
                            var segmentIndex = bi.SegmentFrom(next);
                            if (segmentFromIndex.TryGetValue(segmentIndex, out var ns))
                            {
                                if (ns.items.Length < shortpath)
                                {
                                    targetSegment = ns;
                                    shortpath = ns.items.Length;
                                }
                            }
                        }
                    }
                }

                var newSegment = targetSegment.Append(curr, bi);
                segments.Remove(curr);
                segments.Remove(targetSegment);
                segments.Add(newSegment);
            }

            Console.WriteLine($"\n==========");
            for (int i = 0; i < bi.W; i++)
            {
                Console.Write($"\rwriting {((i) * 100) / bi.W}%        ");
                for (int j = 0; j < bi.H; j++)
                {
                    lb.SetPixel(i, j, bi.pixels[i, j].colorIndex);
                }
            }

            lb.UnlockBits();
            return reduced;
        }
        public static Bitmap GetMap(Bitmap reduced)
        {                     
            var map = new Bitmap(reduced.Width, reduced.Height, PixelFormat.Format8bppIndexed);
            var palette = map.Palette;
            palette.Entries[0] = Color.Black;
            palette.Entries[1] = Color.White;
            palette.Entries[2] = Color.FromArgb(0, 0, 0, 0);
            map.Palette = palette;
            var lb = new LockBitmap(reduced);
            lb.LockBits();
            var bi = new BI { H = (ushort)reduced.Height, pixels = new PI[reduced.Width, reduced.Height], W = (ushort)reduced.Width };
            int totalColors = reduced.Palette.Entries.Length;
            for (int i = 0; i < reduced.Width; i++)
            {
                for (int j = 0; j < reduced.Height; j++)
                {
                    var color = lb.GetPixel(i, j);
                    bi.pixels[i, j] = new PI(i, j, color);
                }
            }
            lb.UnlockBits();
            lb = new LockBitmap(reduced);
            lb.LockBits();

            Console.WriteLine($"\n==========");
            Console.WriteLine($"{totalColors} colors found");
            bi.segmentMap = new int[reduced.Width, reduced.Height];
            List<SI> segments = new List<SI>();
            for (int i = 0; i < reduced.Width; i++)
            {
                for (int j = 0; j < reduced.Height; j++)
                {
                    var segmentIndex = bi.segmentMap[i, j];
                    if (segmentIndex == default)
                    {
                        segmentIndex = segments.Count + 1;
                        Stack<PI> st = new Stack<PI>();
                        List<PI> segmentData = new List<PI>();
                        st.Push(bi.pixels[i, j]);
                        while (st.Count != 0)
                        {
                            var curr = st.Pop();
                            bi.segmentMap[curr.X, curr.Y] = segmentIndex;
                            segmentData.Add(curr);
                            var ok = false;
                            for (var di = -1; di <= 1; di++)
                            {
                                for (var dj = -1; dj <= 1; dj++)
                                {
                                    if (di == 0 && dj == 0)
                                        continue;
                                    var next = curr.Move(di, dj, bi, out ok);
                                    if (ok && next.colorIndex == curr.colorIndex && bi.segmentMap[next.X, next.Y] == default)
                                    {
                                        st.Push(next);
                                    }
                                }
                            }
                        }

                        segments.Add(new SI(segmentData.ToArray(), segmentIndex, segmentData[0].colorIndex));
                        Console.Write($"\r{segmentIndex} segments found");
                    }
                }
            }                        
            lb.UnlockBits();
            lb = new LockBitmap(map);
            lb.LockBits();

            Console.WriteLine($"\n==========");
            foreach (var si in segments.OrderByDescending(x => x.items.Length))
            {
                Console.Write($"{si.items.Length}, ");
            }

            //            Console.WriteLine($"\n==========");
            //            var colorFontValues = Enumerable.Range(0, colors.Count()).ToDictionary(x => x, x => GetColorIndexBytes(font, x+1));
            //            for (int i = 0; i < bi.W; i++) {
            //                for (int j = 0; j < bi.H; j++) {
            //                    lb.SetPixel(i,j, Color.White);
            //                }
            //            }
            //            
            //            for (var index = 0; index < segments.Count; index++) {
            //                Console.Write($"\rprocessing {index} of {segments.Count}");
            //                var si = segments[index];
            //                var sc = si.GetCenter(bi);
            //                var img = colorFontValues[si.colorIndex];
            //                var lt = sc.Move(img.Length/2, 0, bi, out var ltsSuccess);
            //                if (!ltsSuccess)
            //                    lt = bi.pixels[bi.W - 1, lt.Y];
            //                lt = lt.Move(0, img[0].Length/2, bi, out ltsSuccess);
            //                if (!ltsSuccess)
            //                    lt = bi.pixels[lt.X, bi.H-1];
            //                lt = lt.Move(-img.Length, 0, bi, out ltsSuccess);
            //                if (!ltsSuccess)
            //                    lt = bi.pixels[0, lt.Y];
            //                lt = lt.Move(0, -img[0].Length, bi, out ltsSuccess);
            //                if (!ltsSuccess)
            //                    lt = bi.pixels[lt.X, 0];
            //                
            //                var left = (int)lt.X;
            //                for (int i = 0; i < img.Length; i++) {
            //                    left++;
            //                    var top = (int)lt.Y;
            //                    for (int j = 0; j < img[0].Length; j++) {
            //                        top++;
            //
            //                        if (img[i][j])
            //                            lb.SetPixel(left, top, Color.Black);
            //                    }
            //                }
            //            }

            Console.WriteLine($"\n==========");
            for (int i = 0; i < bi.W; i++)
            {
                Console.Write($"\rmap {((i) * 100) / bi.W}%        ");
                for (int j = 0; j < bi.H; j++)
                {
                    var shouldMark = false;
                    if (i == 0 || j == 0 || i == bi.W - 1 || j == bi.H - 1)
                    {
                        shouldMark = true;
                    }
                    else
                    {
                        var curr = bi.segmentMap[i, j];
                        if (curr != bi.segmentMap[i - 1, j] || curr != bi.segmentMap[i, j - 1] || curr != bi.segmentMap[i - 1, j - 1])
                            shouldMark = true;
                    }

                    lb.SetPixel(i, j, shouldMark ? (byte)0 : (byte)1);
                    // else {
                    //     var color = Color.White;
                    //     var cfw = colorFontValues[bi.pixels[i, j].colorIndex];
                    //     if (cfw[i % cfw.Length][j % cfw[0].Length])
                    //         color = Color.Gray;
                    //     lb.SetPixel(i, j, color);
                    // }
                }
            }
            lb.UnlockBits();
            return map;
        }        
        static bool[][] GetColorIndexBytes(string fontName, int value) {
            var bitmap = new Bitmap(fontName);
            var lb = new LockBitmap(bitmap);
            lb.LockBits();
            var w = bitmap.Width / 10;
            var h = bitmap.Height;
            bool[,] bmp = new bool[bitmap.Width, h];
            for (int i = 0; i < bitmap.Width; i++) {
                for (int j = 0; j < h; j++) {
                    bmp[i, j] = lb.GetPixel(i, j) == 0;
                }
            }
            lb.UnlockBits();
            var length = value.ToString().Length;
            var offset = (int)(w * 0.5);
            var arrW = (length) * w + (length - 1) * offset;
            var arrH = (int)(h);
            var result = new bool[arrW][];
            var left = 0;
            for (int i = 0; i < length; i++) {
                var sign = (int)(value.ToString()[i] - '0');
                for (int pi = 0; pi < w; pi++) {
                    result[left + pi] = new bool[arrH];
                    for (int pj = 0; pj < h; pj++) {
                        result[left + pi][pj] = bmp[sign * w + pi, pj];
                    }
                }

                left += offset + w;
            }

            for (int i = 0; i < result.Length; i++) {
                if (result[i] == null)
                    result[i] = new bool[arrH];
            }

            return result;
        }
//        public static void Main(string[] args) {
//            var i = 41;
//            Process(5000,
//                    @"C:\Users\serov.alexey\Documents\print\",
//                    i.ToString(),
//                    (i+1).ToString(),
//                    "font2",
//                    ".bmp",
//                    true);
//        }



        class SI {
            public PI[] items { get; }
            public int index { get; }
            public byte colorIndex { get; }
            public SI(PI[] items, int index, byte colorIndex) {
                this.items = items;
                this.index = index;
                this.colorIndex = colorIndex;
            }
            public SI Append(SI second, BI bitmap) {
                var newItems = new PI[this.items.Length + second.items.Length];
                this.items.CopyTo(newItems, 0);
                var startAt = this.items.Length;
                for (int i = 0; i < second.items.Length; i++) {
                    var px = second.items[i];
                    var npx = new PI(px.X, px.Y, this.colorIndex);
                    bitmap.pixels[px.X, px.Y] = npx;
                    bitmap.segmentMap[px.X, px.Y] = this.index;
                    newItems[startAt + i] = npx;
                }

                return new SI(newItems, index, colorIndex);
            }
            public Rectangle CalcBounds() {
                var leftTop = new Point(int.MaxValue, int.MaxValue);
                var rightBottom = new Point(int.MinValue, int.MinValue);
                for (int i = 0; i < items.Length; i++) {
                    var curr = items[i];
                    if (curr.X < leftTop.X && curr.Y < leftTop.Y) {
                        leftTop = new Point(curr.X, curr.Y);
                    } else if (curr.Y > rightBottom.Y && curr.X > rightBottom.X) {
                        rightBottom = new Point(curr.X, curr.Y);
                    }
                }

                return new Rectangle(leftTop, new Size(rightBottom.X - leftTop.X, rightBottom.Y - leftTop.Y));
            }
            public PI GetCenter(BI bitmap) {
                HashSet<PI> set = new HashSet<PI>(items.Where((x,i)=>i%4==0));
                HashSet<PI> set2;
                PI last = default;
                while (set.Count!=0) {
                    set2 = new HashSet<PI>();
                    foreach (var pi in set) {
                        last = pi;
                        bool add = true;
                        for (int i = -10; i <= 10; i+=10) {
                            if(!add)
                                break;
                            for (int j = -10; j <= 10; j+=10) {
                                if (i == 0 && j == 0)
                                    continue;

                                var next = pi.Move(i, j, bitmap, out add);
                                if (add)
                                    add = set.Contains(next);
                                if(!add)
                                    break;
                            }                    
                        }

                        if (add)
                            set2.Add(pi);
                    }
                    set = set2;
                }

                return last;
            }
        }
        struct BI {
            public PI[,] pixels;
            public int[,] segmentMap;
            public ushort W;
            public ushort H;
            public int SegmentFrom(PI source) {
                return this.segmentMap[source.X, source.Y];
            }
        }
        struct PI : IEquatable<PI> {
            public bool Equals(PI other) {
                return this.X == other.X && this.Y == other.Y;
            }
            public override bool Equals(object obj) {
                return obj is PI other && Equals(other);
            }
            public override int GetHashCode() {
                unchecked {
                    return (this.X.GetHashCode() * 397) ^ this.Y.GetHashCode();
                }
            }
            public static bool operator ==(PI left, PI right) {
                return left.Equals(right);
            }
            public static bool operator !=(PI left, PI right) {
                return !left.Equals(right);
            }
            public ushort X;
            public ushort Y;
            public byte colorIndex;
            public PI(int x, int y, byte colorIndex) {
                this.X = (ushort)x;
                this.Y = (ushort)y;
                this.colorIndex = colorIndex;
            }
            public PI Move(int dX, int dY, BI source, out bool success) {
                if (
                    this.X < -dX ||
                    this.Y < -dY ||
                    this.X >= (source.W - dX) ||
                    this.Y >= (source.H - dY)
                ) {
                    success = false;
                    return this;
                }

                success = true;
                return source.pixels[this.X + dX, this.Y + dY];
            }
        }
    }

    public class LockBitmap {
        Bitmap source = null;
        IntPtr Iptr = IntPtr.Zero;
        BitmapData bitmapData = null;
        Rectangle rect;

        public int PixelCount { get; private set; }
        public byte[] Pixels { get; set; }
        public int Width { get; private set; }
        public int Height { get; private set; }

        public LockBitmap(Bitmap source) {
            this.source = source;
            this.rect = new Rectangle(0,0, source.Width, source.Height);
            Width = source.Width;
            Height = source.Height;
            PixelCount = Width * Height;
        }


        public void LockBits() {
            bitmapData = source.LockBits(rect, ImageLockMode.ReadWrite,
                source.PixelFormat);
            Pixels = new byte[PixelCount];
            Iptr = bitmapData.Scan0;
            Marshal.Copy(Iptr, Pixels, 0, Pixels.Length);
        }

        public void UnlockBits() {
            try {
                // Copy data from byte array to pointer
                Marshal.Copy(Pixels, 0, Iptr, Pixels.Length);

                // Unlock bitmap data
                source.UnlockBits(bitmapData);
            }
            catch (Exception ex) {
                throw ex;
            }
        }
        public byte GetPixel(int x, int y) {
            return Pixels[(y * Width) + x];
        }
        public void SetPixel(int x, int y, byte color) {
            int i = ((y * Width) + x);
            Pixels[i] = color;
        }
    }
}