from model.Transformer import GPT2Model
from dataset import TitleGenDataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import torch

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os


rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))

dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank)

is_main = (rank == 0)
if is_main:
    print(f"Main: world_size: {world_size}")

train_dataset = TitleGenDataset(vocab_path="sohu_data/vocab.txt", data_path="sohu_data/data.json")
sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
dataloader = DataLoader(train_dataset, batch_size=3, sampler=sampler, num_workers=20, pin_memory=True, collate_fn=collate_fn)
model = GPT2Model(24, 8724, 768, 12).cuda()
model = DDP(model, device_ids=[local_rank])
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction='none')



for epoch in range(5):
    sampler.set_epoch(epoch)
    model.train()
    for batch_idx, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].cuda()
        target_ids = batch["target_ids"].cuda()
        pad_mask = batch["pad_mask"].cuda()
        content_lens = batch["content_lens"]

        logits = model(input_ids, content_lens)  # [B, max_len, vocab_size]
        B, max_len, vocab_size = logits.shape

        loss_all = criterion(logits.reshape(B*max_len, vocab_size), target_ids.reshape(B*max_len))
        loss_all = loss_all.reshape(B, max_len)

        title_mask = torch.zeros(B, max_len, dtype=torch.float32, device=logits.device)
        for i in range(B):
            c = content_lens[i]
            title_mask[i, c:] = 1.0

        final_mask = title_mask * pad_mask

        loss = (loss_all * final_mask).sum() / final_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_main and (batch_idx+1) % 100 == 0:
            print(f"Epoch {epoch} batch: {batch_idx+1} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

dist.destroy_process_group()




