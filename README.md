# PyTorch Distributed Training Examples

Practical implementations for mastering distributed LLM training. From basics to production-scale systems.

## 📚 Learning Path

### Week 1-2: PyTorch DDP
- [01_basic_ddp](./01_ddp_basics) - Multi-GPU training fundamentals
- [02_ddp_advanced](./02_ddp_advanced) - Gradient accumulation, mixed precision, profiling

### Week 3-4: PyTorch FSDP
- [03_fsdp_basics](./03_fsdp_basics) - Fully sharded data parallel
- [04_fsdp_advanced](./04_fsdp_advanced) - CPU offloading, auto-wrap policies

### Week 5-6: DeepSpeed
- [05_deepspeed_zero](./05_deepspeed) - ZeRO stage 1/2/3 implementations
- [06_deepspeed_advanced](./06_deepspeed_advanced) - Offloading, 3D parallelism

### Week 7-8: Megatron-LM
- [07_tensor_parallel](./07_megatron) - Tensor parallelism
- [08_pipeline_parallel](./08_pipeline) - Pipeline parallelism, 3D parallelism

### Week 9: NCCL
- [09_nccl_benchmarks](./09_nccl) - Collective operations, profiling

### Week 10: Mixed Precision
- [10_mixed_precision](./10_mixed_precision) - FP16/BF16 training

### Week 11-12: AWS Infrastructure
- [11_aws_infrastructure](./11_aws) - Multi-node training on EC2/SageMaker/Trainium

## 🚀 Quick Start

```bash
# Setup environment
conda create -n llm-training python=3.10 -y
conda activate llm-training
pip install -r requirements.txt

# Run first example
cd 01_ddp_basics
torchrun --nproc_per_node=4 train_ddp.py
```

## 📊 Progress Tracker

| Week | Topic | Status | Throughput | Memory | Blog Post |
|------|-------|--------|------------|--------|-----------|
| 1-2  | DDP   | 🔄     | -          | -      | [Link]()  |
| 3-4  | FSDP  | ⏳     | -          | -      | -         |

## 🎯 Goals

- Train models from 124M to 13B+ parameters
- Master all parallelism strategies
- Achieve >50% MFU on production workloads
- Build portfolio for AWS AGI Senior Research Engineer role

## 📝 Blog Series

Each example includes detailed documentation explaining:
- Theory and architecture
- Implementation details
- Performance benchmarks
- Production best practices
