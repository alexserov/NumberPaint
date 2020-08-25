using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;

namespace Viewer.UI
{
    public class FixedLineWrapPanel : Panel 
    {
        public int LineCount
        {
            get { return (int)GetValue(LineCountProperty); }
            set { SetValue(LineCountProperty, value); }
        }        
        public static readonly DependencyProperty LineCountProperty =
            DependencyProperty.Register("LineCount", typeof(int), typeof(FixedLineWrapPanel), new PropertyMetadata(1));

        protected override Size MeasureOverride(Size availableSize)
        {
            var visibleChildren = Children.OfType<UIElement>().Where(x => x.IsVisible).ToList();
            var measureSize = new Size(availableSize.Width * LineCount / visibleChildren.Count, availableSize.Height / LineCount);
            var elementSize = new Size();
            foreach(UIElement element in visibleChildren)
            {
                element.Measure(measureSize);
                elementSize = new Size(Math.Max(element.DesiredSize.Width, elementSize.Width), Math.Max(element.DesiredSize.Height, elementSize.Height));
            }
            return new Size(elementSize.Width * visibleChildren.Count / LineCount, elementSize.Height * LineCount);
        }
        protected override Size ArrangeOverride(Size finalSize)
        {
            var visibleChildren = Children.OfType<UIElement>().Where(x => x.IsVisible).ToList();
            var vcc = visibleChildren.Count;
            var arrangeSize = new Size(finalSize.Width * LineCount / vcc, finalSize.Height / LineCount);
            for (int j = 0; j < vcc / LineCount; j++) {
                for (int i = 0; i < LineCount; i++) {
                    var childIndex = i+ j * LineCount;
                    if (childIndex >= visibleChildren.Count)
                        goto ArrangeEnd;
                    var child = visibleChildren[childIndex];
                    var rect = new Rect(j * arrangeSize.Width, i * arrangeSize.Height, arrangeSize.Width, arrangeSize.Height);
                    child.Arrange(rect);
                }
            }
        ArrangeEnd:
            return finalSize;
        }
    }
}
