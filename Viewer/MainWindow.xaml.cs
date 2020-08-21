using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
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
            var fileName = this.source.Text;
            using var imageStream = new FileStream(fileName, FileMode.Open);
            var sourceBitmap = new System.Drawing.Bitmap(imageStream);
            this.Source.Source = FromBitmap(sourceBitmap);
            var quantizer = new OctreeQuantizer(int.Parse(this.ColorsCount.Text), int.Parse(this.MaxBpp.Text));
            var ResultBitmap = quantizer.Quantize(sourceBitmap);
            this.Quantized.Source = FromBitmap(ResultBitmap);
            RegionWorker.Process(int.Parse(this.Small.Text), ResultBitmap, out var reduced, out var map);

            this.Simplified.Source = FromBitmap(reduced);
            this.Map.Source = FromBitmap(map);
        }
    }

}