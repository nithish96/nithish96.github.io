
#  Multiprocessing

The Python `multiprocessing` library is a module that supports the spawning of processes using a similar API to the threading module. It allows parallel execution of code on multiple processors or cores, taking advantage of multiple CPU cores to improve the performance of certain types of tasks. Here are the main components of the `multiprocessing` library


### **Process**

- The `Process` class is the fundamental component of the `multiprocessing` library. It is used to create a new process.
- Instances of this class represent a separate process of execution. Each process has its own Python interpreter and memory space.

``` py
from multiprocessing import Process

def my_function():
    print("Hello from a subprocess!")

if __name__ == "__main__":
    my_process = Process(target=my_function)
    my_process.start()
    my_process.join()

```

###  **Pool**
- The `Pool` class provides a simple way to parallelize the execution of a function across multiple input values.
- It represents a pool of worker processes that can be used to parallelize the execution of a function across a set of input values.
- 
``` py
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        result = pool.map(square, [1, 2, 3, 4, 5])
    print(result)

```

### **Queue**

- The `Queue` class is used for communication between processes. It allows data to be exchanged between processes in a thread-safe manner.
- Processes can put items into the queue using the `put()` method and retrieve items using the `get()` method.

``` py 
from multiprocessing import Process, Queue

def worker(q):
    data = q.get()
    print(f"Worker received: {data}")

if __name__ == "__main__":
    my_queue = Queue()
    my_process = Process(target=worker, args=(my_queue,))
    my_queue.put("Hello from the main process!")
    my_process.start()
    my_process.join()

```

###  **Lock**

- The `Lock` class provides a way to synchronize access to shared resources. It allows only one process or thread to access the shared resource at a time.

``` py 
from multiprocessing import Process, Lock

def update_shared_data(lock, shared_data):
    with lock:
        shared_data.value += 1

if __name__ == "__main__":
    from multiprocessing import Value

    shared_data = Value("i", 0)
    my_lock = Lock()

    processes = [Process(target=update_shared_data, args=(my_lock, shared_data)) for _ in range(5)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Final value:", shared_data.value)
	
```
### **Pipe**

- The `Pipe` class creates a two-way communication channel between two processes, allowing them to send and receive messages

``` py 
from multiprocessing import Process, Pipe

def sender(conn):
    conn.send("Hello from the sender!")

def receiver(conn):
    data = conn.recv()
    print(f"Receiver received: {data}")

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    sender_process = Process(target=sender, args=(parent_conn,))
    receiver_process = Process(target=receiver, args=(child_conn,))

    sender_process.start()
    receiver_process.start()

    sender_process.join()
    receiver_process.join()

```
### **Manager**
- The `Manager` class provides a way to create shared objects and data structures that can be accessed by multiple processes.
    
- It's useful for sharing more complex data structures like lists, dictionaries, and custom objects.

``` py
from multiprocessing import Process, Manager

def update_shared_list(shared_list, index):
    shared_list[index] = index * index

if __name__ == "__main__":
    with Manager() as manager:
        my_list = manager.list([0, 0, 0, 0, 0])
        processes = [Process(target=update_shared_list, args=(my_list, i)) for i in range(5)]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        print("Updated shared list:", my_list)

```
### **Event**
- The `Event` is a useful tool for coordinating activities among multiple processes in a multiprocessing environment.
- In the following example, `process1` waits for the event to be set, and `process2` sets the event after a delay.  
``` py 
from multiprocessing import Process, Event
import time

def wait_for_event(event):
    print("Waiting for event to be set.")
    event.wait()
    print("Event has been set. Continuing.")

def set_event(event, delay):
    print(f"Sleeping for {delay} seconds before setting the event.")
    time.sleep(delay)
    event.set()
    print("Event has been set.")

if __name__ == "__main__":
    my_event = Event()

    process1 = Process(target=wait_for_event, args=(my_event,))
    process2 = Process(target=set_event, args=(my_event, 3))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

```


### **Barrier**
- `Barrier` class provides a way to synchronize multiple processes by blocking them until a specified number of processes have reached the barrier.
- Process barriers are useful in scenarios where multiple processes need to synchronize their progress, ensuring that they reach a common point before proceeding further
- In the following example, three worker processes are created, each waiting at the barrier. Once all three processes have reached the barrier, they proceed, and the main process prints a completion message
``` py 
from multiprocessing import Barrier, Process
import time

def worker(barrier):
    print(f"Worker {barrier.n_waiting + 1} is waiting at the barrier.")
    barrier.wait()
    print(f"Worker {barrier.n_waiting + 1} has passed the barrier.")

if __name__ == "__main__":
    num_processes = 3
    my_barrier = Barrier(num_processes)

    processes = [Process(target=worker, args=(my_barrier,)) for _ in range(num_processes)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Main process has completed.")

```

### **Condition**

- The `Condition` class provides a way for multiple processes to synchronize their execution based on a shared condition. It is similar to the `threading.Condition` class in the standard library but designed for use with multiprocessing.

``` py 
from multiprocessing import Process, Condition
import time

def worker(condition):
    with condition:
        print("Worker waiting for a condition.")
        condition.wait()
        print("Worker woke up!")

def notifier(condition):
    with condition:
        print("Notifier notifying the condition.")
        condition.notify_all()

if __name__ == "__main__":
    my_lock = Lock()
    my_condition = Condition(my_lock)

    worker_process = Process(target=worker, args=(my_condition,))
    notifier_process = Process(target=notifier, args=(my_condition,))

    worker_process.start()

    time.sleep(2)  # Simulate some work in the main process

    notifier_process.start()
    notifier_process.join()

    with my_condition:
        print("Main process notifying the condition.")
        my_condition.notify_all()

    worker_process.join()

```

### References 

1. [Multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)
2. [ Python Multiprocessing: The Complete Guide](https://superfastpython.com/multiprocessing-in-python/)