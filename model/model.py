import torch
import torch.nn as nn
import torchvision.models as models

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        scores = self.attn(x)                     # (B, T, 1)
        weights = torch.softmax(scores, dim=1)    # normalize across sequence
        context = torch.sum(weights * x, dim=1)   # weighted sum → (B, H)
        return context

class ResNetBiLSTM(nn.Module):
    def __init__(self, 
                 num_artists,
                 hidden_dim=512, 
                 num_layers=1, 
                 bidirectional=True, 
                 dropout=0.2):
        super(ResNetBiLSTM, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.feature_dim = 2048

        self.lstm = nn.LSTM(
            input_size = self.feature_dim, 
            hidden_size = hidden_dim, 
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0,
            bidirectional = bidirectional
        )

        self.directions = 2 if bidirectional else 1
        self.lstm_output_dim = hidden_dim * self.directions
        
        self.attention = AttentionPooling(self.lstm_output_dim)

        self.classifier = nn.Linear(self.lstm_output_dim, num_artists)


    def forward(self, x):

        features = self.backbone(x)              # (B, 2048, 7, 7)
        B, C, H, W = features.shape

        sequence = features.contiguous().view(B, C, H * W).permute(0, 2, 1)  # (B, T=49, C=2048)

        lstm_out, _ = self.lstm(sequence)        # (B, T, H*D)

        embedding = self.attention(lstm_out)     # (B, H*D)

        artist_logits = self.classifier(embedding)  # (B, num_artists)          
        return {
            "artist": artist_logits,
            "embedding": embedding
        }