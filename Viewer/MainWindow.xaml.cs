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
        
        ImageOperations operations;
        public MainWindow() {
            DataContext = this;
            Operations = new ImageOperations(Dispatcher);            
            Operations.Register<Bitmap>(EditorKeys.Image, ImageOperationName.Open, (operations, previous) => {
                var fileName = Dispatcher.Invoke(() => this.source.Text);
                using var imageStream = new FileStream(fileName, FileMode.Open);
                return new System.Drawing.Bitmap(imageStream);
            }).Register<Bitmap>(EditorKeys.Image, ImageOperationName.Quantize, (operations, previous) => {
                var colorCount = Dispatcher.Invoke(() => int.Parse(this.ColorsCount.Text));
                var maxBits = Dispatcher.Invoke(() => int.Parse(this.MaxBpp.Text));
                var quantizer = new OctreeQuantizer(colorCount, maxBits);
                return quantizer.Quantize(previous);
            }).Register<Bitmap>(EditorKeys.Palette, ImageOperationName.Palette, (operations, previous) => {
                return Dispatcher.Invoke(() => {
                    var colorCount = int.Parse(this.ColorsCount.Text);
                    return previous.Palette.Entries.Take(colorCount).Select(x => new SolidColorBrush(System.Windows.Media.Color.FromRgb(x.R, x.G, x.B)));
                });
            }).Register<Bitmap>(EditorKeys.Image, ImageOperationName.Oilify, (operations, previous) => {
                previous = (Bitmap)operations[ImageOperationName.Quantize];
                var rad = Dispatcher.Invoke(() => int.Parse(this.Oilify.Text));
                var levels = Dispatcher.Invoke(() => int.Parse(this.Levels.Text));
                var cudaOilify = Dispatcher.Invoke(() => cudaOilifyBox.IsChecked == true);
                var colorCount = Dispatcher.Invoke(() => int.Parse(this.ColorsCount.Text));
                new Oilify().Execute(previous, out var OilifiedBitmap, rad, levels, colorCount, cudaOilify);
                return OilifiedBitmap;
            }).Register<Bitmap>(EditorKeys.Image, ImageOperationName.Simplify, (operations, previous) => {
                var small = Dispatcher.Invoke(() => int.Parse(this.Small.Text));
                return RegionWorker.GetRegions(previous, small);
            }).Register<Bitmap>(EditorKeys.Image, ImageOperationName.Map, (operations, previous) => {
                return RegionWorker.GetMap(previous);
            })
            ;
            InitializeComponent();
        }

        unsafe ImageSource FromBitmap(Bitmap source)
        {
            return ImageVisualizer.ImageSourceFromBitmap(source);
        }
        CancellationTokenSource cancellationTokenSource;

        public ImageOperations Operations { get => operations; set => operations = value; }

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

            Operations.Execute();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {

        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            //var simplifiedBitmap = (Bitmap)Simplified.Tag;
            //var result = HeatMap.Build(simplifiedBitmap);
            //var pFrom = (0, 255, 0);
            //var pTo = (255, 0, 0);
            //var mLev = (byte)Math.Min(255, result.levels);
            //var delta = ((pTo.Item1 - pFrom.Item1) / mLev, (pTo.Item2 - pFrom.Item2) / mLev, (pTo.Item3 - pFrom.Item3) / mLev);
            //var res = new byte[result.map.Length*3];
            //for(int i = 0; i<result.map.Length; i++)
            //{
            //    var level = (byte)Math.Min(255, result.map[i]);
            //    res[i * 3 + 0] = (byte)(pFrom.Item1 + delta.Item1 * level);
            //    res[i * 3 + 1] = (byte)(pFrom.Item2 + delta.Item2 * level);
            //    res[i * 3 + 2] = (byte)(pFrom.Item3 + delta.Item3 * level);
            //}
            //var visualizer = new ImageVisualizer(res, simplifiedBitmap.Width, simplifiedBitmap.Height);
            //heatMap.Source = visualizer.ImageSource;
        }
    }

}