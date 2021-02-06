import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # removing the <end> word as it wont be passed as input to network, lstm stops after outputing <end> word
        captions = captions[:, :-1]
        
        captions = self.word_embedding(captions)
        
        x = features.unsqueeze(1)
        
        # concat features and captions so timeseries follows like,
        # feature -> <start> -> expectedword1 -> expectedword2 -> ....
        x = torch.cat((x,captions), 1)
        
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        x = inputs
        
        # find next max_len words (output from lstm)
        for _ in range(max_len):
            lstm_out, states = self.lstm(x, states)
            out = self.linear(lstm_out.squeeze(1))
            out = out.max(1)[1]
            outputs.append(out.item())
            
            # come out of loop if model predicted <end>
            if out==1:
                break
            
            # pass the predicted output as input to model to predict next word
            x = self.word_embedding(out)
            x = x.unsqueeze(1)
        
        return outputs