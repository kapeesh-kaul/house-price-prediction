# Project: House Price Prediction using Autoencoder-LSTM

## Overview
This project predicts house prices for Canadian cities using a hybrid Autoencoder-LSTM model. The workflow includes data preprocessing, dimensionality reduction, sequential modeling, and evaluation. It leverages economic indicators such as the MLS Home Price Index (HPI), Consumer Price Index (CPI), and Prime Rate.

---

## Installation Instructions

### Prerequisites
1. **Python 3.8+**
2. **Conda Package Manager** (recommended for managing dependencies)
3. **Git**

### Steps to Set Up

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Conda Environment**
   Create a new environment using the provided `requirements.yml` file.
   ```bash
   conda env create -f requirements.yml
   conda activate house-price-prediction
   ```

3. **Install Additional Dependencies (if needed)**
   ```bash
   pip install -r additional-requirements.txt  # Use this if any extra dependencies are required
   ```

4. **Check Directory Structure**
   Ensure the following folder structure exists:
   ```plaintext
   .
   ├── data
   │   ├── CPI_MONTHLY.csv
   │   ├── house_price_index.xlsx
   │   ├── Prime-Rate-History.csv
   ├── model
   │   ├── model.py
   │   ├── params.json
   │   ├── utils.py
   ├── preprocessing
   │   ├── load_cpi.py
   │   ├── load_rate.py
   ├── main.py
   ├── requirements.yml
   ```

---

## Usage Instructions

### Data Preprocessing
1. **Prepare Raw Data:**
   Ensure the `data` directory contains the required datasets:
   - `CPI_MONTHLY.csv`
   - `house_price_index.xlsx`
   - `Prime-Rate-History.csv`

2. **Generate Data Loaders:**
   Run the `main.py` script to generate and cache dataloaders for training, validation, and testing.
   ```bash
   python main.py --reload
   ```

### Train the Model
Run the following command to train the model:
```bash
python main.py
```
This script will preprocess the data, initialize the Autoencoder-LSTM model, train it, and log results using MLflow.

### Evaluate the Model
After training, the model is evaluated on the test set. Results, including metrics like `R² score` and `test loss`, will be printed to the console.

---

## File Descriptions

- **`main.py`:** Orchestrates the entire workflow, from preprocessing to model evaluation.
- **`model/`:** Contains the Autoencoder-LSTM model definition and utility functions for training and evaluation.
- **`preprocessing/`:** Scripts for loading and preprocessing data (e.g., CPI, HPI, and Prime Rate).
- **`data/`:** Directory to store raw input data.
- **`requirements.yml`:** Specifies the Conda environment dependencies.

---

## Results
- Training and validation losses, as well as evaluation metrics (e.g., `R² score`), are logged and accessible via MLflow.
- Test Loss: **0.00795**
- R² Score: **0.994**

---

## Additional Notes
1. **Reproducibility:** Ensure MLflow is configured correctly to track experiments.
2. **Customization:** Modify `params.json` to experiment with different hyperparameters (e.g., learning rate, latent space size, LSTM layers).
3. **Future Work:** The model can be extended to other cities or include additional features such as demographic or geographic data.

---

## License
This project is open-source and available under the MIT License.
