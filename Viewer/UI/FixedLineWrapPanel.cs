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


        public bool Square {
            get { return (bool)GetValue(SquareProperty); }
            set { SetValue(SquareProperty, value); }
        }

        // Using a DependencyProperty as the backing store for Square.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty SquareProperty =
            DependencyProperty.Register("Square", typeof(bool), typeof(FixedLineWrapPanel), new PropertyMetadata(false));


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
            if (visibleChildren.Count == 0)
                return new Size();
            var lc = LineCount;
            if (Square) {
                lc = (int)(Math.Ceiling(Math.Sqrt(visibleChildren.Count)));
            }
            var measureSize = new Size(availableSize.Width * lc / visibleChildren.Count, availableSize.Height / lc);
            var elementSize = new Size();
            foreach(UIElement element in visibleChildren)
            {
                element.Measure(measureSize);
                elementSize = new Size(Math.Max(element.DesiredSize.Width, elementSize.Width), Math.Max(element.DesiredSize.Height, elementSize.Height));
            }
            return new Size(elementSize.Width * visibleChildren.Count / lc, elementSize.Height * lc);
        }
        protected override Size ArrangeOverride(Size finalSize)
        {
            var visibleChildren = Children.OfType<UIElement>().Where(x => x.IsVisible).ToList();
            if (visibleChildren.Count == 0)
                return new Size();
            var vcc = visibleChildren.Count;
            var lc = LineCount;
            if (Square) {
                lc = (int)(Math.Ceiling(Math.Sqrt(visibleChildren.Count)));
            }            
            var arrangeSize = new Size(finalSize.Width * lc / vcc, finalSize.Height / lc);
            for (int j = 0; j < vcc / lc; j++) {
                for (int i = 0; i < lc; i++) {
                    var childIndex = i+ j * lc;
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
