using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Viewer {
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        public MainWindow() {
            InitializeComponent();

        }

        ImageSource FromBitmap(Bitmap source) {
            var result = new BitmapImage();
            var ms = new MemoryStream();
            source.Save(ms, ImageFormat.Png);
            ms.Position = 0;
            result.BeginInit();
            result.StreamSource = ms;
            result.EndInit();

            return result;
        }

        void ButtonBase_OnClick(object sender, RoutedEventArgs e) {
            var colorCount = int.Parse(this.ColorsCount.Text);
            var maxBits = int.Parse(this.MaxBpp.Text);
            var fileName = this.source.Text;
            var rad = int.Parse(this.Oilify.Text);
            var small = int.Parse(this.Small.Text);
            var levels = int.Parse(this.Levels.Text);
            this.Source.Source = null;
            this.Quantized.Source = null;
            this.Simplified.Source = null;
            this.Oilified.Source = null;
            this.Map.Source = null;
            Task.Run(() => {
                using var imageStream = new FileStream(fileName, FileMode.Open);
                return new System.Drawing.Bitmap(imageStream);
            }).ContinueWith(x => {
                this.Source.Dispatcher.Invoke(() => { this.Source.Source = FromBitmap(x.Result); });
                return x.Result;
            }).ContinueWith(x => {
                var quantizer = new OctreeQuantizer(colorCount, maxBits);
                return quantizer.Quantize(x.Result);
            }).ContinueWith(x => {
                this.Quantized.Dispatcher.Invoke(() => { this.Quantized.Source = FromBitmap(x.Result); });
                return x.Result;
            }).ContinueWith(x => {
                new Oilify().Execute(x.Result, out var OilifiedBitmap, rad, levels, colorCount);
                return OilifiedBitmap;
            }).ContinueWith(x => {
                this.Oilified.Dispatcher.Invoke(() => { this.Oilified.Source = FromBitmap(x.Result); });
                return x.Result;
            }).ContinueWith(x => {
                RegionWorker.Process(small, x.Result, out var reduced, out var map);
                return new {reduced = reduced, map = map};
            }).ContinueWith(x => {
                this.Oilified.Dispatcher.Invoke(() => {
                    this.Simplified.Source = FromBitmap(x.Result.reduced);
                    this.Map.Source = FromBitmap(x.Result.map);
                });
            });
        }
    }

}