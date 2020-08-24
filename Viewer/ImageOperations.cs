using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Viewer
{
    public class ImageOperations<TKey>
    {
        Dictionary<TKey, Bitmap> results = new Dictionary<TKey, Bitmap>();
        List<ImageOperation> operations = new List<ImageOperation>();
        Dictionary<TKey, ImageOperation> operationsMap = new Dictionary<TKey, ImageOperation>();
        static readonly object synchronized = new object();
        readonly CancellationTokenSource tokenSource = new CancellationTokenSource();
        public ImageOperations()
        {            
        }
        public void Cancel()
        {
            tokenSource.Cancel();
        }
        public void Execute()
        {
            var t = Task.Run(() => this, tokenSource.Token);
            foreach (var operation in EnumerateOperations())
            {
                t = t.ContinueWith(x => {
                    if (x.IsCanceled)
                        throw new TaskCanceledException();
                    return operation.Task(x.Result);
                })
                    .ContinueWith(x =>
                    {
                        if (x.IsCanceled)
                            throw new TaskCanceledException();
                        operation.Result = x.Result;
                        this.SetResult(operation.Key, x.Result);
                        return this;
                    });
            }
        }

        void SetResult(TKey key, Bitmap result)
        {
            lock (synchronized)
            {
                results[key] = result;
            }
        }
        public Bitmap GetResult(TKey key)
        {
            lock (synchronized)
            {
                return results[key];
            }
        }
        public Bitmap this[TKey key]=>GetResult(key);
        public ImageOperations<TKey> Register(TKey key, Func<ImageOperations<TKey>, Bitmap, Bitmap> operation)
        {
            var index = operations.Count - 1;
            operations.Add(new ImageOperation() {  Task = x=>operation(x, index==-1 ? null : x.operations[index].Result), Key = key });
            return this;
        }
        public ImageOperations<TKey> StopOnce(TKey key)
        {
            operationsMap[key].StopOnce();
            return this;
        }
        public ImageOperations<TKey> Stop(TKey key)
        {
            operationsMap[key].Stop();
            return this;
        }
        public ImageOperations<TKey> Start(TKey key)
        {
            operationsMap[key].Start();
            return this;
        }
        public ImageOperations<TKey> ExecuteFrom(TKey key)
        {            
            for(int i = 0; i<operations.Count; i++)
            {
                var current = operations[i];
                if (Equals(current.Key, key))
                    break;
                current.StopOnce();
                
            }
            Execute();
            return this;
        }


        IEnumerable<ImageOperation> EnumerateOperations()
        {
            foreach(var element in operations.ToArray())
            {
                yield return element;
                element.AfterExeucte();
            }
        }
        public class ImageOperation
        {
            public Func<ImageOperations<TKey>, Bitmap> Task { get; set; }
            public TKey Key { get; set; }
            public Bitmap Result { get; set; }
            bool stopped;
            bool once;

            internal void Start()
            {
                stopped = false;
                once = false;
            }

            internal void Stop()
            {
                stopped = true;
                once = false;
            }

            internal void StopOnce()
            {
                stopped = true;
                once &= true;
            }

            internal void AfterExeucte()
            {
                if (once)
                    stopped = false;
                once = false;
            }
        }
    }
}
