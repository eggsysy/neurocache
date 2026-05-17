import torch
import torch.nn as nn
import os

class ContextPredictorLSTM_V3(nn.Module):
    """The exact same V3 architecture from our Kaggle notebook."""
    def __init__(self, num_apps, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):
        super(ContextPredictorLSTM_V3, self).__init__()
        self.embedding = nn.Embedding(num_apps, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_apps)
        self.softmax = nn.Softmax(dim=1) # Need this to convert raw scores to percentages

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return self.softmax(out)

class PredictionEngine:
    """Loads the trained weights and runs real inference."""
    def __init__(self, apps_list, model_path="simulator/neurocache_model_v3.pth"):
        self.apps_list = apps_list
        self.num_apps = len(apps_list)
        self.app_to_idx = {app: i for i, app in enumerate(apps_list)}
        self.idx_to_app = {i: app for i, app in enumerate(apps_list)}
        
        # Initialize the V3 Model
        self.model = ContextPredictorLSTM_V3(self.num_apps)
        
        # Load the trained weights from Kaggle
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval() # Lock the model into evaluation mode
            print(f"[AI ENGINE] Successfully loaded V3 weights from {model_path}")
        else:
            print(f"[WARNING] Could not find {model_path}. Model will output random garbage.")

    def get_predictions(self, recent_history_apps):
        """
        Takes the sequence of apps used and asks the Neural Network what happens next.
        """
        try:
            seq_indices = [self.app_to_idx[app] for app in recent_history_apps]
        except KeyError:
            return {app: 0.01 for app in self.apps_list}
            
        # Convert to PyTorch tensor
        input_tensor = torch.tensor([seq_indices], dtype=torch.long)
        
        # Run inference!
        with torch.no_grad():
            output_probs = self.model(input_tensor)[0]
            
        # Convert tensor probabilities back to a dictionary
        predictions = {}
        for idx, prob in enumerate(output_probs):
            predictions[self.idx_to_app[idx]] = round(prob.item(), 3)
            
        return predictions