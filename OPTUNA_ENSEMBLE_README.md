# Sleep Apnea Detection - Optuna Integration & Ensemble Methods

This implementation adds hyperparameter optimization using Optuna and ensemble methods to combine the best ML and DL models for sleep apnea detection.

## New Features

### 1. Hyperparameter Optimization with Optuna

Optuna is integrated to automatically find the best hyperparameters for both ML and DL models.

#### Supported Models

**Machine Learning Models:**
- SVM (Support Vector Machine)
- Random Forest
- Gradient Boosting
- XGBoost

**Deep Learning Models:**
- CNN1D (1D Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- CNN_LSTM (Hybrid CNN-LSTM)

#### Search Spaces

The hyperparameter search spaces are defined in `src/config.py`:

**ML Models:**
- SVM: C, gamma, kernel
- Random Forest: n_estimators, max_depth, min_samples_split
- Gradient Boosting: n_estimators, learning_rate, max_depth, subsample
- XGBoost: n_estimators, learning_rate, max_depth, subsample, colsample_bytree

**DL Models:**
- CNN1D: dropout, learning_rate, batch_size
- LSTM: hidden_size, num_layers, dropout, learning_rate, batch_size
- CNN_LSTM: lstm_hidden, lstm_layers, dropout, learning_rate, batch_size

### 2. Ensemble Methods

Multiple ensemble strategies are implemented to combine the predictions of multiple models:

#### Voting Ensemble
Combines predictions through majority voting (hard) or probability averaging (soft).

#### Weighted Ensemble
Assigns weights to each model based on their validation performance.

#### Stacking Ensemble
Uses a meta-learner (Logistic Regression) to combine base model predictions.

#### Hybrid Ensemble
Combines ML and DL models, handling their different input formats (features vs. raw data).

### 3. Model Selection

Automatically selects the top N models based on validation performance for ensemble creation.

## Usage

### Basic Usage (Default Hyperparameters)

Run with default hyperparameters:

```bash
python main.py --quick-test
```

### With Hyperparameter Optimization

Enable Optuna for hyperparameter tuning:

```bash
python main.py --use-optuna --optuna-trials 50 --quick-test
```

**Parameters:**
- `--use-optuna`: Enable Optuna hyperparameter optimization
- `--optuna-trials`: Number of optimization trials (default: 50)

### With Ensemble Methods

Create and evaluate ensemble models:

```bash
python main.py --use-ensemble --quick-test
```

### Combined: Optuna + Ensemble

Use both hyperparameter optimization and ensemble methods:

```bash
python main.py --use-optuna --optuna-trials 30 --use-ensemble --quick-test
```

### ML or DL Only

Run only ML or DL models:

```bash
# ML only with Optuna and ensemble
python main.py --ml-only --use-optuna --use-ensemble --quick-test

# DL only with Optuna
python main.py --dl-only --use-optuna --optuna-trials 20 --epochs 30 --quick-test
```

### Full Pipeline

Run the complete pipeline with all features:

```bash
python main.py --use-optuna --optuna-trials 50 --use-ensemble --epochs 50
```

**Note:** Remove `--quick-test` for full dataset training.

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ml-only` | Run only ML models | False |
| `--dl-only` | Run only DL models | False |
| `--epochs` | Number of epochs for DL training | 50 |
| `--use-test-set` | Use x01-x35 as separate test set | False |
| `--quick-test` | Quick test with limited data | False |
| `--use-optuna` | Enable Optuna hyperparameter optimization | False |
| `--optuna-trials` | Number of Optuna trials | 50 |
| `--use-ensemble` | Create and evaluate ensemble models | False |

## Configuration

Edit `src/config.py` to customize:

- `OPTUNA_N_TRIALS`: Default number of trials (50)
- `OPTUNA_TIMEOUT`: Timeout in seconds (None for no timeout)
- `ML_PARAM_SPACE`: Hyperparameter search spaces for ML models
- `DL_PARAM_SPACE`: Hyperparameter search spaces for DL models
- `ENSEMBLE_TOP_N_MODELS`: Number of top models for ensemble (3)
- `ENSEMBLE_VOTING_TYPE`: Voting type ('soft' or 'hard')

## Implementation Details

### Module Structure

```
src/
├── config.py                    # Configuration with search spaces
├── hyperparameter_tuning.py     # Optuna optimization
├── ensemble.py                  # Ensemble methods
├── ml_models.py                 # ML model implementations
├── dl_models.py                 # DL model implementations
└── ...
```

### Hyperparameter Tuning Workflow

1. **Define Search Space**: Hyperparameter ranges in `config.py`
2. **Optimize**: Optuna runs trials to find best parameters
3. **Train**: Models are retrained with optimized parameters
4. **Evaluate**: Performance on test set

### Ensemble Workflow

1. **Train Models**: Individual models are trained
2. **Select Top Models**: Best models selected by validation AUC
3. **Create Ensemble**: Combine using voting, weighting, or stacking
4. **Evaluate**: Ensemble performance on test set

## Example Output

```
================================================================
ML HYPERPARAMETER OPTIMIZATION (OPTUNA)
================================================================

Optimizing RandomForest hyperparameters...
Best RandomForest AUC: 0.9234
Best parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5}

================================================================
ENSEMBLE MODELS
================================================================

Creating ML Voting Ensemble...
  Voting Ensemble - Accuracy: 0.9145, AUC: 0.9456

Creating ML Weighted Ensemble...
  Weighted Ensemble - Accuracy: 0.9178, AUC: 0.9489

Creating Hybrid Ensemble (ML + DL)...
  Hybrid Ensemble - Accuracy: 0.9312, AUC: 0.9567
```

## Performance Tips

1. **Start Small**: Use `--quick-test` to verify the pipeline works
2. **Tune Trials**: Start with 10-20 trials, increase for better results
3. **GPU for DL**: Ensure CUDA is available for faster DL training
4. **Parallel Optimization**: Use `OPTUNA_N_JOBS` in config for parallel trials
5. **Monitor**: Check Optuna progress bars and validation scores

## Dependencies

New dependencies added to `requirements.txt`:
- `optuna>=3.0.0` - Hyperparameter optimization

## Notes

- Hyperparameter optimization can take significant time
- Use `--quick-test` for rapid iteration and testing
- Ensemble methods work best with diverse models
- Hybrid ensembles combine strengths of ML (features) and DL (raw signals)
- Results are saved to `results/` directory with timestamps

## Troubleshooting

**Issue**: Optuna optimization is slow  
**Solution**: Reduce `--optuna-trials` or use `--quick-test`

**Issue**: Out of memory during DL optimization  
**Solution**: Reduce batch size in DL_PARAM_SPACE or use CPU

**Issue**: Ensemble not improving performance  
**Solution**: Ensure base models are diverse and well-performing

## Future Improvements

- [ ] Bayesian optimization for search space
- [ ] Multi-objective optimization (accuracy + inference time)
- [ ] AutoML pipeline selection
- [ ] Distributed training for large-scale optimization
