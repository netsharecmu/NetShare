# DDP development checklist

## Toy dataset (pcap), single node single GPU

- [x] DDP = false, ray = false, n_chunk=1
- [x] DDP = true, ray = false, n_chunk=1
- [x] DDP = false, ray = false, n_chunk=2
- [x] DDP = true, ray = false, n_chunk=2

## Toy dataset (pcap), single node 2 GPUs

- [x] DDP = true, ray = false, n_chunk=1
- [x] DDP = true, ray = false, n_chunk=2
- [x] DDP = true, ray = false, n_chunk=3
- [x] DDP = true, ray = false, n_chunk=4
