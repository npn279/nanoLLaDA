import os
import pickle
import torch
import argparse
from model import GPT2
from generate import generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running dLLM pretraining")
    parser.add_argument("--path", type=str, default="pretrained_gpt_last.pt")
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
    model.eval()

    ######################################################
    ################ LOADING PRETRAINED MODEL ############
    ######################################################
    filename = args.path 
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint)

    ######################################################
    ################ GENERATION HYPERPARAMETERS ##########
    ######################################################
    temperature, cfg_scale, gen_length, steps = 0.0, 0.0, 128, 128

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
