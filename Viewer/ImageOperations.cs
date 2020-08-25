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
    }
    public class ImageOperations<TKey> {
        public IEnumerable<IImageOperation> OperationsSource { get { return operations; } }
        List<ImageOperation> operations = new List<ImageOperation>();
        Dictionary<TKey, ImageOperation> operationsMap = new Dictionary<TKey, ImageOperation>();
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
        public object GetResult(TKey key) {
            lock (synchronized) {
                return operationsMap[key].Result;
            }
        }
        public object this[TKey key] => GetResult(key);
        public ImageOperations<TKey> Register<TValueIn>(TKey key, Func<ImageOperations<TKey>, TValueIn, object> operation) {
            var index = operations.Count - 1;
            operations.Add(new ImageOperation(uiDispatcher) { Task = x => operation(x, index == -1 ? default(TValueIn) : (TValueIn)x.operations[index].Result), Key = key });
            return this;
        }
        public ImageOperations<TKey> StopOnce(TKey key) {
            operationsMap[key].StopOnce();
            return this;
        }
        public ImageOperations<TKey> Stop(TKey key) {
            operationsMap[key].Stop();
            return this;
        }
        public ImageOperations<TKey> Start(TKey key) {
            operationsMap[key].Start();
            return this;
        }
        public ImageOperations<TKey> ExecuteFrom(TKey key) {
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
            public Func<ImageOperations<TKey>, object> Task { get; set; }
            public TKey Key { get; set; }
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

            public ImageOperation(Dispatcher uiDispatcher) {
                this.uiDispatcher = uiDispatcher;
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
            
            PropertyChangedEventHandler propertyChanged = (_, __) => { };
            private object result;

            event PropertyChangedEventHandler INotifyPropertyChanged.PropertyChanged {
                add { propertyChanged += value; }
                remove { propertyChanged -= value; }
            }
        }
    }
}