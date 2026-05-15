from argparse import ArgumentParser
import torch


def main():
    parser = ArgumentParser()
    parser.add_argument("train_pt")
    parser.add_argument("model_ckpt")
    parser.add_argument("dest")
    args = parser.parse_args()

    train_sd = torch.load(args.train_pt, map_location="cpu", weights_only=True)
    model_ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    model_sd = model_ckpt.state_dict()

    if not set(train_sd.keys()) == set(model_sd.keys()):
        raise RuntimeError()
    for key in train_sd.keys():
        train_val = train_sd[key]
        model_val = model_sd[key]
        if not train_val.shape == model_val.shape:
            raise RuntimeError()

    model_ckpt.load_state_dict(train_sd)
    torch.save(model_ckpt, args.dest)

if __name__ == "__main__":
    main()
