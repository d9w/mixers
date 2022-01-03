import argparse
import torch

from model.wide_res_net import WideResNet
from model.mlp_mixer import MLPMixer
from model.convmixer import ConvMixer
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.smooth_cross_entropy import smooth_crossentropy
from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop")
    parser.add_argument("--patch", default=32, type=int, help="Patch size")
    parser.add_argument("--hidden", default=512, type=int, help="Size of hidden_dim")
    parser.add_argument("--token", default=256, type=int, help="Size of tokens_hidden_dim")
    parser.add_argument("--channels", default=2048, type=int, help="Size of channels_hidden_dim")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay")
    parser.add_argument("--data_dir", default="./data", type=str, help="CIFAR storage directory")
    parser.add_argument("--checkpoint", default="checkpoint.model", type=str, help="Model checkpoint file")
    parser.add_argument("--conv", default=False, type=bool, help="True for ConvMixer, False for MLPMixer")
    parser.add_argument("--kernel", default=8, type=int, help="Size of conv kernel")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads, data_dir=args.data_dir)
    log = Log(log_each=10)
    val_acc = 0
    # model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    if args.conv:
        model = ConvMixer(args.hidden, args.depth, kernel_size=args.kernel,
                          patch_size=args.patch, n_classes=10).to(device)
    else:
        model = MLPMixer(num_classes=10, image_size=32, patch_size=args.patch,
                         hidden_dim=args.hidden, tokens_hidden_dim=args.token,
                         channels_hidden_dim=args.channels, num_layers=args.depth).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
                    lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
            if log.best_accuracy > val_acc:
                val_acc = log.best_accuracy
                torch.save(model.state_dict(), args.checkpoint)

    log.flush()
