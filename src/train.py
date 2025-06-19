import os
import pickle

import torch
import argparse
from model import GPT2
from misc import train_model
from generate import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running dLLM pretraining")
    parser.add_argument("--save_filename", type=str, default='pretrained_gpt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######################################################
    ######## CREATING TOKEN ENCODER AND DECODER ##########
    ######################################################

    data_dir = "data/shakespeare_char/"
    meta_path = os.path.join(data_dir, "meta.pkl")
    vocab_size = None

    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        itos = meta["itos"]
        stoi = meta["stoi"]
        # adding MASK token
        itos[vocab_size] = "[MASK]"
        stoi["[MASK]"] = vocab_size
        vocab_size += 1
        print(f"found vocab_size = {vocab_size} (inside {meta_path})")

    encode = lambda s: torch.tensor([stoi[c] for c in s])
    decode = lambda l: "".join([itos[i] for i in l])

    ######################################################
    ################ GPT2 HYPERPARAMETERS ################
    ######################################################
    block_size = 512
    n_layer, n_head, n_embd, dropout = 4, 8, 256, 0.1
    model = GPT2(
        vocab_size=vocab_size,
        block_size=block_size,
        dim_emb=n_embd,
        n_heads=n_head,
        n_layers=n_layer,
        flash=True,
    )
    model = model.to(device)
    print("GPT number of parameters:", sum(p.numel() for p in model.parameters()))

    ######################################################
    ################ TRAINING HYPERPARAMETERS ############
    ######################################################
    batch_size, max_iters, lr, weight_decay, grap_clip, mask_id = (
        64,
        1000,
        1e-3,
        1e-7,
        0.1,
        vocab_size - 1,
    )
    model, losses, val_losses = train_model(model, data_dir, batch_size, max_iters, lr, weight_decay, grap_clip, mask_id, device=device, save_filename=args.save_filename)
   
    ######################################################
    ################ GENERATION HYPERPARAMETERS ##########
    ######################################################
    temperature, cfg_scale, gen_length, steps = 0.0, 0.0, 128, 128
    model.eval()

    shakpeare_lines = "Would"
    x = encode(shakpeare_lines)
    x = x.reshape(1, -1)
    x = x.to(device)

    out = generate(
        model,
        x,
        steps=steps,
        gen_length=gen_length,
        block_length=gen_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking="random",
        mask_id=vocab_size - 1,
        device=device,
    )

    generation = decode(out[0].tolist())
    print(generation)
    print("------------------------")
