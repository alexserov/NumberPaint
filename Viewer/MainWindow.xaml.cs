using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using PixelFormat = System.Windows.Media.PixelFormat;

namespace Viewer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

        }

        unsafe ImageSource FromBitmap(Bitmap source)
        {
            var pf = PixelFormats.Bgra32;
            switch (source.PixelFormat)
            {
                case System.Drawing.Imaging.PixelFormat.Format8bppIndexed:
                    pf = PixelFormats.Indexed8;
                    break;
                case System.Drawing.Imaging.PixelFormat.Format24bppRgb:
                    pf = PixelFormats.Bgr24;
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
        CancellationTokenSource cancellationTokenSource;
        void ButtonBase_OnClick2(object sender, RoutedEventArgs e)
        {
            cancellationTokenSource.Cancel();
            executeButton.IsEnabled = true;
            cancelButton.IsEnabled = false;
        }
        void BuildTaskChain()
        {

        }
        enum TaskName
        {

        }
        void ButtonBase_OnClick(object sender, RoutedEventArgs e)
        {
            executeButton.IsEnabled = false;
            cancelButton.IsEnabled = true;
            cancellationTokenSource = new CancellationTokenSource();
            var colorCount = int.Parse(this.ColorsCount.Text);
            var maxBits = int.Parse(this.MaxBpp.Text);
            var fileName = this.source.Text;
            var rad = int.Parse(this.Oilify.Text);
            var small = int.Parse(this.Small.Text);
            var levels = int.Parse(this.Levels.Text);
            var cudaOilify = cudaOilifyBox.IsChecked == true;
            this.Source.Source = null;
            this.Quantized.Source = null;
            this.Simplified.Source = null;
            this.Oilified.Source = null;
            this.Map.Source = null;
            Task.Run(() =>
            {
                using var imageStream = new FileStream(fileName, FileMode.Open);
                return new System.Drawing.Bitmap(imageStream);
            }, cancellationTokenSource.Token).ContinueWith(x =>
            {
                this.Source.Dispatcher.Invoke(() => { this.Source.Source = FromBitmap(x.Result); });
                return x.Result;
            }).ContinueWith(x =>
            {
                var quantizer = new OctreeQuantizer(colorCount, maxBits);
                return quantizer.Quantize(x.Result);
            }).ContinueWith(x =>
            {
                this.Source.Dispatcher.Invoke(() =>
                {
                    OutPalette.ItemsSource = x.Result.Palette.Entries.Take(colorCount).Select(x => new SolidColorBrush(System.Windows.Media.Color.FromRgb(x.R, x.G, x.B)));
                });
                return x.Result;
            }).ContinueWith(x =>
            {
                this.Quantized.Dispatcher.Invoke(() => { this.Quantized.Source = FromBitmap(x.Result); });
                return x.Result;
            }).ContinueWith(x =>
            {
                new Oilify().Execute(x.Result, out var OilifiedBitmap, rad, levels, colorCount, cudaOilify);
                return OilifiedBitmap;
            }).ContinueWith(x =>
            {
                this.Oilified.Dispatcher.Invoke(() => { this.Oilified.Source = FromBitmap(x.Result); });
                return x.Result;
            }).ContinueWith(x =>
            {
                RegionWorker.Process(small, x.Result, out var reduced, out var map);
                return (reduced: reduced, map: map);
            }).ContinueWith(x =>
            {
                this.Oilified.Dispatcher.Invoke(() =>
                {
                    this.Simplified.Source = FromBitmap(x.Result.reduced);
                    this.Map.Source = FromBitmap(x.Result.map);
                    executeButton.IsEnabled = true;
                    cancelButton.IsEnabled = false;
                });
            });
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {

        }
    }

}