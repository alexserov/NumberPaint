using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace Viewer {
    public interface IImageOperation : INotifyPropertyChanged {
        string Header { get; }
        object Result { get; }
        EditorKeys Editor { get; }
    }
    public enum ImageOperationName {
        Unused,
        Open,
        Quantize,
        Palette,
        Oilify,
        Simplify,
        Map,
    }
    public enum EditorKeys {
        Image,
        Palette
    }
    public class ImageOperations {
        public IEnumerable<IImageOperation> OperationsSource { get { return operations; } }
        List<ImageOperation> operations = new List<ImageOperation>();
        Dictionary<ImageOperationName, ImageOperation> operationsMap = new Dictionary<ImageOperationName, ImageOperation>();
        static readonly object synchronized = new object();
        readonly CancellationTokenSource tokenSource = new CancellationTokenSource();
        private readonly Dispatcher uiDispatcher;

        public ImageOperations(Dispatcher uiDispatcher) {
            this.uiDispatcher = uiDispatcher;
        }
        public void Cancel() {
            tokenSource.Cancel();
        }
        public void Execute() {
            var t = Task.Run(() => this, tokenSource.Token);
            foreach (var operation in EnumerateOperations()) {
                t = t.ContinueWith(x => {
                    if (x.IsCanceled)
                        throw new TaskCanceledException();
                    return operation.Task(x.Result);
                })
                    .ContinueWith(x => {
                        if (x.IsCanceled)
                            throw new TaskCanceledException();
                        this.SetResult(operation, x.Result);
                        return this;
                    });
            }
        }

        void SetResult(ImageOperation op, object result) {
            lock (synchronized) {
                op.Result = result;
            }
        }
        public object GetResult(ImageOperationName key) {
            lock (synchronized) {
                return operationsMap[key].Result;
            }
        }
        public object this[ImageOperationName key] => GetResult(key);
        public ImageOperations Register<TValueIn>(EditorKeys editor, ImageOperationName key, Func<ImageOperations, TValueIn, object> operation) {
            var index = operations.Count - 1;
            var op =
            new ImageOperation(uiDispatcher, editor) {
                Task = x => {
                    TValueIn previous = default;
                    if (index != -1) {
                        var prevResult = x.operations[index].Result;
                        if (prevResult is TValueIn)
                            previous = (TValueIn)prevResult;
                    }
                    return operation(x, previous);
                }, Key = key
            };
            operations.Add(op);
            operationsMap[key] = op;
            return this;
        }
        public ImageOperations StopOnce(ImageOperationName key) {
            operationsMap[key].StopOnce();
            return this;
        }
        public ImageOperations Stop(ImageOperationName key) {
            operationsMap[key].Stop();
            return this;
        }
        public ImageOperations Start(ImageOperationName key) {
            operationsMap[key].Start();
            return this;
        }
        public ImageOperations ExecuteFrom(ImageOperationName key) {
            for (int i = 0; i < operations.Count; i++) {
                var current = operations[i];
                if (Equals(current.Key, key))
                    break;
                current.StopOnce();

            }
            Execute();
            return this;
        }


        IEnumerable<ImageOperation> EnumerateOperations() {
            foreach (var element in operations.ToArray()) {
                yield return element;
                element.AfterExeucte();
            }
        }
        public class ImageOperation : IImageOperation {
            public Func<ImageOperations, object> Task { get; set; }
            public ImageOperationName Key { get; set; }
            public object Result {
                get => result; set {
                    if (Equals(result, value))
                        return;
                    result = value;
                    uiDispatcher.Invoke(() => propertyChanged(this, new PropertyChangedEventArgs(nameof(Result))));
                }
            }
            bool stopped;
            bool once;
            private Dispatcher uiDispatcher;

            public ImageOperation(Dispatcher uiDispatcher, EditorKeys editor) {
                this.uiDispatcher = uiDispatcher;
                Editor = editor;
            }

            internal void Start() {
                stopped = false;
                once = false;
            }

            internal void Stop() {
                stopped = true;
                once = false;
            }

            internal void StopOnce() {
                stopped = true;
                once &= true;
            }

            internal void AfterExeucte() {
                if (once)
                    stopped = false;
                once = false;
            }

            public string Header => Key.ToString();

            public EditorKeys Editor { get; }

            PropertyChangedEventHandler propertyChanged = (_, __) => { };
            private object result;

            event PropertyChangedEventHandler INotifyPropertyChanged.PropertyChanged {
                add { propertyChanged += value; }
                remove { propertyChanged -= value; }
            }
        }
    }
}