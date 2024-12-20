from preprocessing import generate_dataloader, save_dataloaders, load_dataloaders
import torch
import argparse
from model import AutoencoderLSTM, ModelParameters,train, evaluate

def show_first_five_entries(dataloader, name):
        print(f"First 5 entries of {name}:")
        for i, data in enumerate(dataloader):
            if i >= 5:
                break
            inputs, targets = data
            print(f"Input: {inputs}, Target: {targets}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    cityname = "Regina"
    parser = argparse.ArgumentParser(description="House Price Prediction")
    parser.add_argument('--reload', action='store_true', help='Flag to load dataloaders from disk')
    args = parser.parse_args()

    if not args.reload:
        print("Loading from cached dataloaders")
        train_loader, val_loader, test_loader = load_dataloaders(cityname)
    else:
        print("Generating new dataloaders")
        train_loader, val_loader, test_loader = generate_dataloader(cityname)
        save_dataloaders(cityname, train_loader, val_loader, test_loader)

    # return

    input_dim = len(next(iter(train_loader))[0][0])
    params = ModelParameters.from_json("model/params.json")
    params.input_dim = input_dim
    model = AutoencoderLSTM(params=params).to(device)

    train_losses, val_losses, r2_scores = train(model, train_loader, val_loader, num_epochs=50, lr=0.001, device=device)

    # Test model
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()