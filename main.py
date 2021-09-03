import torch
import torchvision

def main():

    epsilon = 8/255

    # model
    from utils.model import get_model
    repo_or_dir = "chenyaofo/pytorch-cifar-models" 
    model_name = "cifar10_resnet20"
    model = get_model(repo_or_dir=repo_or_dir, model=model_name)
    model.eval()


    from attacker.PGD import PGDAttacker
    attacker = PGDAttacker()
    attacker.build(
        epsilon=epsilon, model=model, criterion="CrossEnropy",
        norm="Linf", device_ids=[0],
        eta=epsilon, iteration=100
    )

    # dataset
    dataset = torchvision.datasets.CIFAR10(
        root="./", download=True, train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, 
        num_workers=1
    )

    for x, y in dataloader:
        attacker.attack(x_true=x, y_true=y)


if __name__=="__main__":
    main()
