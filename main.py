from preprocessing import generate_dataloader, save_dataloaders, load_dataloaders
import torch
import argparse
from model import AutoencoderLSTM, ModelParameters, train, evaluate
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import logging

# Set the logging level to suppress warnings
logging.getLogger("mlflow").setLevel(logging.ERROR)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    input_dim = len(next(iter(train_loader))[0][0])
    params = ModelParameters.from_json("model/params.json")
    params.input_dim = input_dim
    model = AutoencoderLSTM(params=params).to(device)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("AutoEncoderLSTM")    

    with mlflow.start_run(run_name="AutoEncoderLSTM"):
        mlflow.log_param("cityname", cityname)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("encoded_dim", params.encoded_dim)
        mlflow.log_param("lstm_hidden_dim", params.lstm_hidden_dim)
        mlflow.log_param("lstm_num_layers", params.lstm_num_layers)
        mlflow.log_param("output_dim", params.output_dim)
        mlflow.log_param("num_epochs", 50)
        mlflow.log_param("learning_rate", 0.001)

        train_losses, val_losses, r2_scores = train(model, train_loader, val_loader, num_epochs=50, lr=0.001, device=device)

        for epoch, (train_loss, val_loss, r2_score) in enumerate(zip(train_losses, val_losses, r2_scores)):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("r2_score", r2_score, step=epoch)

        # Test model
        r2, test_loss = evaluate(model, test_loader, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("r2_score", r2)
        print(f"Test Loss: {test_loss:.4f} | R2 Score: {r2:.4f}")

        mlflow.pytorch.log_model(model, "model", conda_env='requirements.yml')

if __name__ == "__main__":
    main()