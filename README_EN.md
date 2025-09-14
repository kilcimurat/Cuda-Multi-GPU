# CudaMultiGPU Usage Documentation

This document explains step by step how to use the **CudaMultiGPU** class in your projects.

---

## 1. Add the Class to Your Project

Copy the `gpu_utils.py` file into your project root directory.  
Import the class in your code:

<img width="468" height="45" alt="image" src="https://github.com/user-attachments/assets/98f5ad01-c035-44d1-b97a-30abb18883ab" />

```python
from gpu_utils import CudaMultiGPU
```

---

## 2. Remove DEVICE Definition

Remove the `DEVICE` definition from your code.

<img width="468" height="46" alt="image" src="https://github.com/user-attachments/assets/96a89fe6-c05b-49d7-94e3-b2fe274fe525" />

---

## 3. Wrapping the Model

To use your existing PyTorch model with multi-GPU support:

<img width="468" height="51" alt="image" src="https://github.com/user-attachments/assets/bacfc348-752e-4fc6-9bd8-7fbaa56a91fc" />

```python
model = MyModel()              # your existing PyTorch model
model = CudaMultiGPU(model)    # detects all CUDA devices and wraps with DataParallel
DEVICE = model.device          # device where the data will be placed
```

- If there is no CUDA device, the model will automatically run on **CPU**.  
- Since the `CudaMultiGPU` instance is callable, it can be used as:  

```python
outputs = model(input_tensor)
```

- To directly access the underlying model:

```python
model.model
```

---

## 4. Setting Up the Optimizer

Since `CudaMultiGPU` forwards attributes and methods to the underlying model, parameters can be accessed directly:

<img width="468" height="28" alt="image" src="https://github.com/user-attachments/assets/d590b533-1c83-4bc5-95c7-63ec67d224b5" />

```python
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
```

---

## 5. Changes in the Training Loop

<img width="468" height="77" alt="image" src="https://github.com/user-attachments/assets/259c951c-3875-4c90-8e6b-7934eb350319" />

### Moving Data to the Correct Device
```python
inputs  = inputs.to(DEVICE)
targets = targets.to(DEVICE)
```

### Calling the Model
```python
outputs = model(inputs)
```

### Loss Averaging
`DataParallel` may return separate loss values for each GPU.  
You should average them for backpropagation:

```python
loss = outputs.loss.mean()  # or loss.mean() depending on your setup
```

### Backpropagation and Optimization
```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

## 6. Adjusting Batch Size and Num Workers

<img width="297" height="152" alt="image" src="https://github.com/user-attachments/assets/057fd002-eb19-4de8-bf8b-2bb5f9dbe1f0" />

Depending on the number of GPUs and the workload capacity, `batch_size` can be increased to **128–256**.  

The recommended calculation for **num_workers** is:

```
num_workers = number of CPU cores / number of GPUs
```

### Practical Suggestions:
- 16-core CPU and 3 GPUs → `num_workers = 4–6`  
- 32-core CPU and 3 GPUs → `num_workers = 8–12`  
- 64-core CPU (Threadripper / EPYC) → `num_workers = 16+`  

These numbers can be tuned further by monitoring GPU utilization.

---

## License
This document and the `CudaMultiGPU` class can be freely used in your projects.
