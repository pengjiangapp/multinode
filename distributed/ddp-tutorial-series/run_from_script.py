import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='script to run training job')
    parser.add_argument('--mode', type=str, help='Total epochs to train the model')
    args = parser.parse_args()

    mode = args.mode
    if mode == 'multigpu':
        os.system("torchrun --standalone --nproc_per_node=4 distributed/ddp-tutorial-series/multigpu_torchrun.py  --total_epochs=50 --save_every=10 --batch_size=32")
    print("===finish training job===")