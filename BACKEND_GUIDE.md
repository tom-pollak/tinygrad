# Adding a New Backend to tinygrad

This guide explains how to implement a new hardware accelerator backend for tinygrad.

## Core Components

To implement a new backend, you need to create four key components:

1. **Renderer** - Converts operations to backend-specific code
2. **Allocator** - Manages memory on your device
3. **Program** - Executes compiled code on your device
4. **Device** - Coordinates the above components

## Step 1: Create Your Backend File

Create a new file in `tinygrad/runtime/` named `ops_<yourbackend>.py` (lowercase).

## Step 2: Implement the Four Core Components

### 2.1 Renderer

The Renderer class converts tinygrad's operations into code for your device:

```python
class YourBackendRenderer(Renderer):  # or extend ClangRenderer for C-based backends
    device = "YOURBACKEND"  # Must match your device name (uppercase)
    
    # Override methods or properties to customize code generation
    # For C-based backends, you might customize:
    buffer_suffix = " __attribute__((aligned(16)))"  # Add alignment or other qualifiers
    kernel_prefix = "__attribute__((noinline)) "     # Add prefixes to kernel functions
    
    # Define mapping of tinygrad ops to your device's code
    code_for_op = {
        Ops.ADD: lambda x, y, dtype: f"({x} + {y})",
        Ops.MUL: lambda x, y, dtype: f"({x} * {y})",
        Ops.SIN: lambda x, dtype: f"sin({x})",
        Ops.LOG2: lambda x, dtype: f"log2({x})",
        Ops.EXP2: lambda x, dtype: f"exp2({x})",
        # Implement all required ops...
    }
```

### 2.2 Allocator

The Allocator manages memory allocation, deallocation, and transfers:

```python
class YourBackendAllocator(Allocator):
    def __init__(self, device):
        self.device = device
        super().__init__()
    
    def _alloc(self, size, options):
        # Allocate memory on your device
        # Return an opaque handle to the allocated memory
        
    def _free(self, opaque, options):
        # Free memory on your device
        
    def _copyin(self, dest, src):
        # Copy data from host (src) to device (dest)
        
    def _copyout(self, dest, src):
        # Copy data from device (src) to host (dest)
        
    def _offset(self, buf, size, offset):
        # Create a view of buffer at offset
        # Required for buffer views
```

### 2.3 Program

The Program class manages executing compiled code:

```python
class YourBackendProgram:
    def __init__(self, device, name, lib):
        self.device = device
        self.lib = lib  # The compiled code
    
    def __call__(self, *bufs, vals=(), wait=False):
        # Execute the program on your device
        # bufs: buffers to read/write
        # vals: scalar values to pass
        # wait: whether to wait for completion
        
        # Your implementation here:
        # 1. Prepare arguments
        # 2. Execute the code
        # 3. Return execution time
        
        return execution_time  # Return execution time in seconds
```

### 2.4 Device

The Device class ties everything together:

```python
class YOURBACKENDDevice(Compiled):  # Class name must be <UPPERCASE>Device
    def __init__(self, device=""):
        # Initialize your device
        try:
            # Setup device-specific resources
            
            # Create instances of your components
            allocator = YourBackendAllocator(self)
            renderer = YourBackendRenderer()
            compiler = Compiler("compile_yourbackend", [...your compiler args...])
            
            # Initialize the base class
            super().__init__(device, allocator, renderer, compiler, YourBackendProgram)
            
        except Exception as e:
            # Fallback to a mock implementation if hardware is not available
            super().__init__(device, MallocAllocator, MockRenderer(), MockCompiler(), MockProgram)
    
    def synchronize(self):
        # Synchronize the device (wait for all operations to complete)
        pass
    
    def finalize(self):
        # Clean up resources when the device is being destroyed
        pass
```

## Step 3: Register Your Backend

Add your device to `ALL_DEVICES` in `tinygrad/device.py`:

```python
ALL_DEVICES = ["METAL", "AMD", "NV", "CUDA", "QCOM", "GPU", "CPU", "LLVM", "DSP", "WEBGPU", "YOURBACKEND"]
```

## Core Operations

Your renderer must support several categories of operations from the `Ops` enum. Here's a comprehensive list of operations that need to be implemented:

### Memory Operations

These operations handle memory access and buffer management:

```python
# Memory operations
Ops.LOAD:            # Load values from memory
Ops.STORE:           # Store values to memory
Ops.INDEX:           # Create an index for memory access
Ops.DEFINE_GLOBAL:   # Define a global buffer
Ops.DEFINE_LOCAL:    # Define local memory
Ops.DEFINE_ACC:      # Define an accumulator
```

### Binary Arithmetic Operations

These handle basic arithmetic between two operands:

```python
# Binary arithmetic
Ops.ADD:    lambda x, y, dtype: f"({x} + {y})"       # Addition
Ops.MUL:    lambda x, y, dtype: f"({x} * {y})"       # Multiplication
Ops.SUB:    lambda x, y, dtype: f"({x} - {y})"       # Subtraction
Ops.IDIV:   lambda x, y, dtype: f"({x} / {y})"       # Integer division
Ops.FDIV:   lambda x, y, dtype: f"({x} / ({y}))"     # Floating-point division
Ops.MOD:    lambda x, y, dtype: f"({x} % {y})"       # Modulo
Ops.POW:    lambda x, y, dtype: f"pow({x}, {y})"     # Power
Ops.MAX:    lambda x, y, dtype: f"max({x}, {y})"     # Maximum
```

### Comparison Operations

These handle comparisons between values:

```python
# Comparison operations
Ops.CMPLT:  lambda x, y, dtype: f"({x} < {y})"       # Less than
Ops.CMPNE:  lambda x, y, dtype: f"({x} != {y})"      # Not equal
```

### Bitwise Operations

These handle bit-level operations:

```python
# Bitwise operations
Ops.AND:    lambda x, y, dtype: f"({x} & {y})"       # Bitwise AND
Ops.OR:     lambda x, y, dtype: f"({x} | {y})"       # Bitwise OR
Ops.XOR:    lambda x, y, dtype: f"({x} ^ {y})"       # Bitwise XOR
Ops.SHL:    lambda x, y, dtype: f"({x} << {y})"      # Shift left
Ops.SHR:    lambda x, y, dtype: f"({x} >> {y})"      # Shift right
Ops.THREEFRY: lambda x, y, dtype: f"threefry({x}, {y})" # Random number generation
```

### Unary Operations

These operate on a single input:

```python
# Unary operations
Ops.NEG:    lambda x, dtype: f"(-{x})"               # Negation
Ops.EXP2:   lambda x, dtype: f"exp2({x})"            # 2^x
Ops.LOG2:   lambda x, dtype: f"log2({x})"            # Log base 2
Ops.SIN:    lambda x, dtype: f"sin({x})"             # Sine
Ops.SQRT:   lambda x, dtype: f"sqrt({x})"            # Square root
Ops.RECIP:  lambda x, dtype: f"(1.0/({x}))"          # Reciprocal (1/x)
```

### Ternary Operations

These operate on three inputs:

```python
# Ternary operations
Ops.WHERE:  lambda x, y, z, dtype: f"({x} ? {y} : {z})"  # Conditional selection
Ops.MULACC: lambda x, y, z, dtype: f"({x} * {y} + {z})"  # Multiply-accumulate
```

### Data Manipulation Operations

These handle data movement and transformation:

```python
# Data manipulation
Ops.CAST:       # Type conversion
Ops.BITCAST:    # Reinterpret bits as different type
Ops.VECTORIZE:  # Combine elements into a vector
Ops.GEP:        # Get element pointer (extract from vector)
```

### Control Flow Operations

These handle execution flow:

```python
# Control flow
Ops.RANGE:      # Loop iteration
Ops.ENDRANGE:   # End of loop
Ops.BARRIER:    # Synchronization point
Ops.IF:         # Conditional branch
Ops.ENDIF:      # End of conditional
```

### Constants and Variables

These define values:

```python
# Constants and variables
Ops.CONST:      # Scalar constant
Ops.VCONST:     # Vector constant
Ops.DEFINE_VAR: # Define a variable
Ops.SPECIAL:    # Special values (thread IDs, etc)
```

### Implementation Note

While some specialized backends might implement additional operations (like tensor core operations such as WMMA), the operations listed above form the core set that any backend needs to handle to support basic tinygrad functionality.

## Complete Example: Python Backend

Here's a complete working implementation of a Python backend:

```python
from typing import Any, Optional, Callable, Union, Tuple
import pickle, base64, itertools, time, struct
from tinygrad.dtype import DType, dtypes
from tinygrad.device import Compiled, Compiler, Allocator, BufferSpec
from tinygrad.ops import exec_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer
from tinygrad.helpers import dtypes_from_groups

def _load(m, i):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]

def load(inp, j=0):
  if len(inp) == 3: return [_load(m, x+j if x is not None else None) if gate else default for (m,x),default,gate in zip(*inp)]
  return [_load(m, x+j if x is not None else None) for m,x in inp[0]]

def _store(m, i, v):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

# This implements most operations needed for a basic backend
class SimplePythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: list[tuple[Ops, Optional[DType], list[int], Any]] = pickle.loads(lib)
    
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    
    # Simple single-threaded execution
    ul: dict[int, Any] = {}  # Maps UOp index to its value
    dl: dict[int, DType] = {}  # Maps UOp index to its dtype
    pbufs: list[memoryview] = list(bufs)  # Buffer arguments
    pvals: list[int] = list(vals)  # Value arguments
    
    i = 0
    loop_ends: dict[int, int] = {}  # For RANGE operation
    
    while i < len(self.uops):
      uop, dtype, idp, arg = self.uops[i]
      # Get inputs from upstream operations
      void_ops = {Ops.STORE, Ops.ENDRANGE, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.NAME}
      inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
      dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]
      
      # Process the operation
      dl[i] = dtype
      
      if uop is Ops.STORE:
        # Store requires special handling
        if len(inp) == 2: inp.append([True] * len(inp[0]))  # set the gate to True
        for (m,o),v,g in zip(*inp):
          if g: _store(m, o, v)
      
      elif uop is Ops.DEFINE_GLOBAL:
        # Get buffer from arguments
        assert dtype.fmt is not None, "DEFINE_GLOBAL requires a format"
        ul[i] = [pbufs.pop(0).cast(dtype.fmt)] if pbufs else [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(dtype.fmt)]
        
      elif uop is Ops.DEFINE_LOCAL:
        # Local memory allocation
        assert dtype.fmt is not None, "DEFINE_LOCAL requires a format"
        ul[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(dtype.fmt)]
        
      elif uop is Ops.DEFINE_ACC:
        # Accumulator initialization
        src_val = 0
        if len(inp) > 0 and len(inp[0]) > 0: 
            src_val = inp[0][0]
        if dtype.count > 1:
            ul[i] = [[src_val] for _ in range(dtype.count)]
        else:
            ul[i] = [src_val]
        
      elif uop is Ops.DEFINE_VAR:
        # Get value from arguments
        ul[i] = [pvals.pop(0) if pvals else 0]
        
      elif uop is Ops.CONST:
        # Constant value
        ul[i] = [arg]
        
      elif uop is Ops.VCONST:
        # Vector constant
        ul[i] = arg
        
      elif uop is Ops.SPECIAL:
        # Special values like global/local IDs
        if arg.startswith('g'): ul[i] = [0]  # Just return 0 for global ID
        elif arg.startswith('l'): ul[i] = [0]  # Just return 0 for local ID
        
      elif uop is Ops.LOAD:
        # Load from memory
        ul[i] = load(inp)
        
      elif uop is Ops.CAST:
        # Type casting
        if dtype.count > 1:
            # Vector cast
            ul[i] = [dtypes.as_const(x, dtype.scalar()) for x in inp[0]]
        else:
            # Scalar cast
            ul[i] = [dtypes.as_const(x, dtype) for x in inp[0]]
            
      elif uop is Ops.INDEX:
        # Create index tuple for memory access
        ret = []
        for m,o in zip(inp[0], inp[1]): 
            ret.append((m,o))
        ul[i] = ret
        
      elif uop is Ops.GEP:
        # Get element pointer - extract elements from vectors
        if isinstance(arg, tuple):
            ul[i] = [inp[0][j] for j in arg]
        else:
            ul[i] = inp[0][arg]
            
      elif uop is Ops.ENDRANGE:
        # End of a range loop
        loop_ends[idp[0]] = i
        i = idp[0]
        continue
        
      elif uop is Ops.RANGE:
        # Range loop
        if i not in ul:
            ul[i] = [inp[0][0]]
        else:
            ul[i][0] += 1
            if ul[i][0] == inp[1][0]:
                del ul[i]
                i = loop_ends[i] + 1
                continue
                
      elif uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.NAME):
        # These are control ops we can ignore in the simple implementation
        pass
        
      elif uop is Ops.VECTORIZE:
        # Create a vector from elements
        ul[i] = inp
        
      elif uop in GroupOp.ALU:
        # Execute all ALU operations
        # Make sure all inputs have same length
        if any(len(x) != len(inp[0]) for x in inp):
            # Broadcast scalars if needed
            inp = [[x[0]]*len(inp[0]) if len(x) == 1 else x for x in inp]
        
        import math
        # Implement all ALU operations directly with match statement
        match uop:
            # Binary arithmetic ops
            case Ops.ADD:
                ul[i] = [a + b for a, b in zip(inp[0], inp[1])]
            case Ops.MUL:
                ul[i] = [a * b for a, b in zip(inp[0], inp[1])]
            case Ops.SUB:
                ul[i] = [a - b for a, b in zip(inp[0], inp[1])]
            case Ops.IDIV:
                ul[i] = [a // b for a, b in zip(inp[0], inp[1])]
            case Ops.FDIV:
                ul[i] = [a / b for a, b in zip(inp[0], inp[1])]
            case Ops.MAX:
                ul[i] = [max(a, b) for a, b in zip(inp[0], inp[1])]
            case Ops.MOD:
                ul[i] = [a % b for a, b in zip(inp[0], inp[1])]
            case Ops.POW:
                ul[i] = [a ** b for a, b in zip(inp[0], inp[1])]
                
            # Comparison ops
            case Ops.CMPLT:
                ul[i] = [a < b for a, b in zip(inp[0], inp[1])]
            case Ops.CMPNE:
                ul[i] = [a != b for a, b in zip(inp[0], inp[1])]
                
            # Bitwise ops    
            case Ops.AND:
                ul[i] = [a & b for a, b in zip(inp[0], inp[1])]
            case Ops.OR:
                ul[i] = [a | b for a, b in zip(inp[0], inp[1])]
            case Ops.XOR:
                ul[i] = [a ^ b for a, b in zip(inp[0], inp[1])]
            case Ops.SHL:
                ul[i] = [a << b for a, b in zip(inp[0], inp[1])]
            case Ops.SHR:
                ul[i] = [a >> b for a, b in zip(inp[0], inp[1])]
            case Ops.THREEFRY:
                # Simple pseudo-random number implementation
                ul[i] = [(a + b*12345) % 0xFFFFFFFF for a, b in zip(inp[0], inp[1])]
                
            # Unary ops
            case Ops.EXP2:
                ul[i] = [2.0 ** a for a in inp[0]]
            case Ops.LOG2:
                ul[i] = [math.log2(max(a, 1e-10)) for a in inp[0]]
            case Ops.SIN:
                ul[i] = [math.sin(a) for a in inp[0]]
            case Ops.SQRT:
                ul[i] = [math.sqrt(max(a, 0)) for a in inp[0]]
            case Ops.RECIP:
                ul[i] = [1.0 / max(a, 1e-10) for a in inp[0]]
            case Ops.NEG:
                ul[i] = [-a for a in inp[0]]
                
            # Ternary ops
            case Ops.WHERE:
                ul[i] = [b if a else c for a, b, c in zip(inp[0], inp[1], inp[2])]
            case Ops.MULACC:
                ul[i] = [a * b + c for a, b, c in zip(inp[0], inp[1], inp[2])]
                
            # Default case
            case _:
                # For any unexpected op, just pass the first input through
                ul[i] = inp[0] if inp else [0]
      else:
          # Any unhandled op gets a default value
          ul[i] = [0]
      
      i += 1
      
    return time.perf_counter() - st

class SimplePythonRenderer(Renderer):
  device = "SIMPLE_PYTHON"
  
  # Define tensor cores (not used in simple implementation but needed for compatibility)
  tensor_cores = []
  
  def render(self, uops:list[UOp]) -> str:
    # Convert UOps to a format our program can execute
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class SimplePythonCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    # Simply decode the serialized operations
    return base64.b64decode(src)

class SimplePythonAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec):
    # Allocate memory using a standard bytearray
    return memoryview(bytearray(size))
    
  def _free(self, opaque, options:BufferSpec):
    # Nothing to do for Python memory
    pass
    
  def _copyin(self, dest, src:memoryview):
    # Copy from host to device
    dest[:] = src
    
  def _copyout(self, dest:memoryview, src):
    # Copy from device to host
    dest[:] = src
    
  def _offset(self, buf, size:int, offset:int):
    # Create a view into the buffer at the given offset
    return buf[offset:offset+size]

class SIMPLE_PYTHONDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(
        device, 
        SimplePythonAllocator(), 
        SimplePythonRenderer(), 
        SimplePythonCompiler(), 
        SimplePythonProgram
    )
    
  def synchronize(self):
    # Nothing to synchronize in our Python implementation
    pass
```

## Testing Your Backend

Once implemented, test your backend:

```python
# Check if your device is recognized
python -c "from tinygrad import Device; print(list(Device.get_available_devices()))"

# Force use of your backend
YOURBACKEND=1 python -c "from tinygrad import Tensor; x = Tensor([1,2,3]); print(x+x)"
```

Remember to add your backend to `ALL_DEVICES` in `tinygrad/device.py` for it to be properly discovered.