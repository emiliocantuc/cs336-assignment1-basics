import torch

from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimization import AdamW, gradient_clipping, CosAnnealingLRScheduler, cross_entropy

import argparse
import os
from functools import partial
import numpy as np
from einops import rearrange


@torch.no_grad()
def evaluate(model, batch_getter, num_batches):
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        input_ids, target_ids = batch_getter()
        logits = model(input_ids)
        loss = cross_entropy(rearrange(logits, "b s c -> (b s) c"), rearrange(target_ids, "b s -> (b s)"))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def train(
    model, optimizer, lr_scheduler, batch_getter, max_iters, log_interval, eval_interval, save_interval, checkpoint_path
):
    for iter in range(max_iters):
        input_ids, target_ids = batch_getter()

        logits = model(input_ids)
        loss = cross_entropy(rearrange(logits, "b s c -> (b s) c"), rearrange(target_ids, "b s -> (b s)"))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        if iter % log_interval == 0:
            print(f"iter {iter}: loss {loss.item():.4f}, lr {lr_scheduler.last_lr:.6f}")

        if iter % eval_interval == 0 and iter > 0:
            val_loss = evaluate(model, get_val_batch, num_batches=100)
            print(f"Validation loss at iteration {iter}: {val_loss:.4f}")

        if iter % save_interval == 0 and iter > 0:
            save_checkpoint(model, optimizer, iter, checkpoint_path)
            print(f"Checkpoint saved at iteration {iter}")

    print("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--theta", type=float, default=10_000.0)

    # data
    parser.add_argument("--train_path", type=str, default="data/train.npy")
    parser.add_argument("--val_path", type=str, default="data/val.npy")

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=8)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=int, default=1_000)
    parser.add_argument("--max_iters", type=int, default=10_000)
    # other
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).to(device)

    optimizer = AdamW(model.parameters())
    lr_scheduler = CosAnnealingLRScheduler(
        optimizer,
        min_lr=args.min_learning_rate,
        max_lr=args.max_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.max_iters,
    )

    if os.path.exists(args.checkpoint_path) and not args.overwrite:
        print(f"Resuming from checkpoint {args.checkpoint_path}")
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        lr_scheduler.t = start_iter
    else:
        lr_scheduler.step()  # set initial lr

    # TODO
    # train_ds = np.memmap(args.train_path, mode="r", dtype=np.uint16)
    # val_ds = np.memmap(args.val_path, mode="r", dtype=np.uint16)
    train_ds = np.random.randint(0, args.vocab_size, size=100_000, dtype=np.uint16)[:-1]
    val_ds = train_ds[1:]

    get_train_batch = partial(
        get_batch,
        x=train_ds,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )

    get_val_batch = partial(
        get_batch,
        x=val_ds,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )

    # if args.wandb:
    #     import wandb

    #     wandb.init(project="cs336-basics", config=vars(args))
    #     wandb.watch(model, log="all")

    train(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_getter=get_train_batch,
        max_iters=args.max_iters,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_path=args.checkpoint_path,
    )
