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
using System.Windows.Media.Media3D;
using PixelFormat = System.Windows.Media.PixelFormat;

namespace Viewer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        enum ImageOperationName
        {
            Unused,
            OpenFile,
            SetSource,
            Quantize,
            SetQuantizedPalette,
            SetQuantized,
            Oilify,
            SetOilified,
            GetSimplified,
            GetMap,
            SetSimplified,
            SetMap
        }
        ImageOperations<ImageOperationName> operations;
        public MainWindow()
        {         
            operations = new ImageOperations<ImageOperationName>();
            InitializeComponent();
            operations.Register(ImageOperationName.OpenFile, (operations, previous) =>
            {
                var fileName = Dispatcher.Invoke(() => this.source.Text);
                using var imageStream = new FileStream(fileName, FileMode.Open);
                return new System.Drawing.Bitmap(imageStream);
            }).Register(ImageOperationName.SetSource, (operations, previous) =>
            {
                Dispatcher.Invoke(() => { this.Source.Source = FromBitmap(previous); });
                return previous;
            }).Register(ImageOperationName.Quantize, (operations, previous) =>
            {
                var colorCount = Dispatcher.Invoke(() => int.Parse(this.ColorsCount.Text));
                var maxBits = Dispatcher.Invoke(() => int.Parse(this.MaxBpp.Text));
                var quantizer = new OctreeQuantizer(colorCount, maxBits);
                return quantizer.Quantize(previous);
            }).Register(ImageOperationName.SetQuantizedPalette, (operations, previous) =>
            {

                Dispatcher.Invoke(() =>
                {
                    var colorCount = int.Parse(this.ColorsCount.Text);
                    OutPalette.ItemsSource = previous.Palette.Entries.Take(colorCount).Select(x => new SolidColorBrush(System.Windows.Media.Color.FromRgb(x.R, x.G, x.B)));
                });
                return previous;
            }).Register(ImageOperationName.SetQuantized, (operations, previous) =>
            {
                Dispatcher.Invoke(() => { this.Quantized.Source = FromBitmap(previous); this.Quantized.Tag = previous; });
                return previous;
            }).Register(ImageOperationName.Oilify, (operations, previous) =>
            {
                var rad = Dispatcher.Invoke(() => int.Parse(this.Oilify.Text));
                var levels = Dispatcher.Invoke(() => int.Parse(this.Levels.Text));
                var cudaOilify = Dispatcher.Invoke(() => cudaOilifyBox.IsChecked == true);
                var colorCount = Dispatcher.Invoke(() => int.Parse(this.ColorsCount.Text));
                new Oilify().Execute(previous, out var OilifiedBitmap, rad, levels, colorCount, cudaOilify);
                return OilifiedBitmap;
            }).Register(ImageOperationName.SetOilified, (operations, previous) =>
            {
                Dispatcher.Invoke(() => { this.Oilified.Source = FromBitmap(previous); this.Oilified.Tag = previous; });
                return previous;
            }).Register(ImageOperationName.GetSimplified, (operations, previous) =>
            {
                var small = Dispatcher.Invoke(() => int.Parse(this.Small.Text));
                return RegionWorker.GetRegions(previous, small);
            }).Register(ImageOperationName.SetSimplified, (operations, previous) =>
            {
                Dispatcher.Invoke(() => { this.Simplified.Source = FromBitmap(previous); });
                return previous;
            }).Register(ImageOperationName.GetMap, (operations, previous) =>
            {
                return RegionWorker.GetMap(previous);
            }).Register(ImageOperationName.SetMap, (operations, previous) =>
            {
                Dispatcher.Invoke(() =>
                {
                    this.Map.Source = FromBitmap(previous);
                });
                return previous;
            });
        }

        unsafe ImageSource FromBitmap(Bitmap source)
        {
            return ImageVisualizer.ImageSourceFromBitmap(source);
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
        void ButtonBase_OnClick(object sender, RoutedEventArgs e)
        {
            executeButton.IsEnabled = false;
            cancelButton.IsEnabled = true;
            cancellationTokenSource = new CancellationTokenSource();
            
            this.Source.Source = null;
            this.Quantized.Source = null;
            this.Simplified.Source = null;
            this.Oilified.Source = null;
            this.Map.Source = null;

            operations.Execute();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {

        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            var simplifiedBitmap = (Bitmap)Simplified.Tag;
            var result = HeatMap.Build(simplifiedBitmap);
            var pFrom = (0, 255, 0);
            var pTo = (255, 0, 0);
            var mLev = (byte)Math.Min(255, result.levels);
            var delta = ((pTo.Item1 - pFrom.Item1) / mLev, (pTo.Item2 - pFrom.Item2) / mLev, (pTo.Item3 - pFrom.Item3) / mLev);
            var res = new byte[result.map.Length*3];
            for(int i = 0; i<result.map.Length; i++)
            {
                var level = (byte)Math.Min(255, result.map[i]);
                res[i * 3 + 0] = (byte)(pFrom.Item1 + delta.Item1 * level);
                res[i * 3 + 1] = (byte)(pFrom.Item2 + delta.Item2 * level);
                res[i * 3 + 2] = (byte)(pFrom.Item3 + delta.Item3 * level);
            }
            var visualizer = new ImageVisualizer(res, simplifiedBitmap.Width, simplifiedBitmap.Height);
            heatMap.Source = visualizer.ImageSource;
        }
    }

}