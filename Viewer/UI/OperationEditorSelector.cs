using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;

namespace Viewer.UI {
    public class OperationEditorSelector : DataTemplateSelector {
        public DataTemplate ImageTemplate { get; set; }
        public DataTemplate PaletteTemplate { get; set; }
        public override DataTemplate SelectTemplate(object item, DependencyObject container) {
            var operation = (IImageOperation)item;
            switch (operation.Editor) {
                case EditorKeys.Image:
                    return ImageTemplate;                    
                case EditorKeys.Palette:
                    return PaletteTemplate;
                default:
                    throw new InvalidOperationException();
            }
            return base.SelectTemplate(item, container);
        }
    }
}
