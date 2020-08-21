using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using AVNC.Classes;

namespace Viewer {
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        ImageProcessor processor = new ImageProcessor();
        public MainWindow() {
            InitializeComponent();
            this.ImagesBox.ItemsSource = GetSampleResourceNames();

        }
        public IEnumerable<string> GetSampleResourceNames() {
            return typeof(MainWindow).Assembly.GetManifestResourceNames();
        }

        void ImagesBox_OnSelectionChanged(object sender, SelectionChangedEventArgs e) {
            this.processor.ResourceName = ((ListBox) sender).SelectedItem as string;
            var imageStream = typeof(MainWindow).Assembly.GetManifestResourceStream(this.processor.ResourceName);
            this.processor.SourceBitmap = new System.Drawing.Bitmap(imageStream);
            this.processor.SourceImage = new BitmapImage();
            this.processor.SourceImage.BeginInit();
            imageStream.Seek(0, SeekOrigin.Begin);
            this.processor.SourceImage.StreamSource = imageStream;
            this.processor.SourceImage.EndInit();
            this.processor.SourceImage.Freeze();
            this.Source.Source = this.processor.SourceImage;
        }

        void ButtonBase_OnClick(object sender, RoutedEventArgs e) {
            this.processor.ProcessImage();
            this.Result.Source = this.processor.ResultImage;
        }
    }

    public class ImageProcessor {
        public string ResourceName { get; set; }
        public System.Drawing.Bitmap SourceBitmap { get; set; }
        public System.Drawing.Bitmap ResultBitmap { get; set; }
        public BitmapImage SourceImage { get; set; }
        public BitmapImage ResultImage { get; set; }
        
        public ImageProcessor() { }
        
        public void ProcessImage() {
            var quantizer = new OctreeQuantizer(16, 8);
            ResultBitmap = quantizer.Quantize(SourceBitmap);
            
            ResultImage = new BitmapImage();
            var ms = new MemoryStream();
            ResultBitmap.Save(ms, ImageFormat.Png);
            ms.Position = 0;
            ResultImage.BeginInit();
            ResultImage.StreamSource = ms;
            ResultImage.EndInit();
        }
    }
    
}