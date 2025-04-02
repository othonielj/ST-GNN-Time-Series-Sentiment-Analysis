# ST-GNN Time Series Sentiment Analysis

A sophisticated stock market analysis system that combines Graph Neural Networks (GNN) with time series analysis and sentiment analysis to predict stock price movements.

## Overview

This project implements a Spatio-Temporal Graph Neural Network (ST-GNN) that analyzes:
- Historical stock price data
- Market sentiment from news articles
- Inter-stock relationships
- Technical indicators

The system provides:
- Stock price predictions
- Sentiment analysis
- Market trend analysis
- Individual stock performance metrics

## Features

- **Data Collection & Processing**
  - Automated stock data retrieval
  - News article collection and sentiment analysis
  - Technical indicator calculation
  - Data normalization and preprocessing

- **Graph Neural Network**
  - Spatio-temporal graph construction
  - Multi-head attention mechanism
  - Dynamic edge weight calculation
  - Temporal pattern recognition

- **Sentiment Analysis**
  - News article collection
  - Topic modeling
  - Sentiment scoring
  - Impact analysis

- **Prediction & Visualization**
  - Stock price predictions
  - Performance metrics
  - Interactive visualizations
  - Model evaluation tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/othonielj/ST-GNN-Time-Series-Sentiment-Analysis.git
cd ST-GNN-Time-Series-Sentiment-Analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── A_data_loader.py      # Data collection and preprocessing
├── B_fetchnews.py        # News article collection
├── C_sentiment_analysis.py  # Sentiment analysis
├── D_preprocess_stock_data.py  # Stock data preprocessing
├── E_graph_builder.py    # Graph construction
├── F_visualize_graph.py  # Graph visualization
├── G_exp_dat_analysis.py # Exploratory data analysis
├── H_model.py           # Model architecture
├── I_train.py           # Training pipeline
├── J_evaluate.py        # Model evaluation
├── K_predict.py         # Prediction generation
├── L_utils.py           # Utility functions
├── M_config.py          # Configuration settings
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Usage

1. **Data Collection**
```bash
python A_data_loader.py  # Collect stock data
python B_fetchnews.py    # Collect news articles
```

2. **Data Processing**
```bash
python C_sentiment_analysis.py  # Analyze sentiment
python D_preprocess_stock_data.py  # Preprocess stock data
```

3. **Model Training**
```bash
python I_train.py  # Train the ST-GNN model
```

4. **Prediction**
```bash
python K_predict.py  # Generate predictions
```

5. **Evaluation**
```bash
python J_evaluate.py  # Evaluate model performance
```

## Configuration

Edit `M_config.py` to configure:
- Stock tickers
- Date ranges
- Model parameters
- Data paths
- API keys (if needed)

## Dependencies

- Python 3.8+
- PyTorch
- PyTorch Geometric
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- transformers
- nltk

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue in the GitHub repository. 