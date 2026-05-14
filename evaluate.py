"""
Gradescope evaluation script.
Usage: python evaluate.py --model_path checkpoints/best_model.pt
"""
import argparse
import torch
from src.data      import get_dataloaders
from src.model     import Transformer
from src.inference import evaluate_bleu


def load_model(model_path, device):
    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    cfg       = ckpt['cfg']
    src_vocab = ckpt['src_vocab']
    tgt_vocab = ckpt['tgt_vocab']

    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg['d_model'],
        num_heads      = cfg['num_heads'],
        num_layers     = cfg['num_layers'],
        d_ff           = cfg['d_ff'],
        max_len        = cfg['max_len'],
        dropout        = cfg['dropout'],
        pos_encoding   = cfg.get('pos_encoding', 'sinusoidal'),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.set_vocabs(src_vocab, tgt_vocab)
    return model, src_vocab, tgt_vocab, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoints/best_model.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, src_vocab, tgt_vocab, cfg = load_model(args.model_path, device)
    print(f"Model loaded. Testing infer() with a sample string...")

    # Quick sanity check
    sample = "ein hund läuft über das gras ."
    result = model.infer(sample)
    print(f"  DE: {sample}")
    print(f"  EN: {result}")

    _, _, test_loader, _, _ = get_dataloaders(
        batch_size  = 64,
        max_len     = cfg['max_len'],
        min_freq    = cfg['min_freq'],
        num_workers = 0,
    )

    bleu = evaluate_bleu(model, test_loader, src_vocab, tgt_vocab, device)
    print(f"Test BLEU: {bleu:.2f}")


if __name__ == '__main__':
    main()
