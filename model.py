
  
import torch 
import torch.nn as nn
from torch.autograd import Variable
import math

# Hyper-parameters
imsize = 1, 650, 650
batch_size = 28
sequence_length = 41**2 #convolved size
dim = hidden_size = 128
num_layers = 1
num_classes = alphabet = 100 #100 ascii printable
num_epochs = 25


# Device configuration
# scaler = torch.cuda.amp.GradScaler() #fp16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
	"Convolutional layers. 2d represenation -> 2d map"
	def __init__(self, in_channels, pool_height):
		super(Encoder, self).__init__()
		self.pool_height = pool_height
		self.in_channels = in_channels
		self.sigmoid    = nn.Sigmoid()
		self.tanh       = nn.Tanh()
		self.pool       = nn.MaxPool2d(kernel_size=(2, 2))
		self.conv1      = nn.Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3)
		self.conv2      = nn.Conv2d(8, 16,  (2,4),  1, (1, 2))
		self.conv2_gate = nn.Conv2d(16, 16,  3,     1,  1)
		self.conv3      = nn.Conv2d(16, 32,  (2,4), 1, (1, 2))
		self.conv3_gate = nn.Conv2d(32, 32,  3,     1,  1)
		self.conv4      = nn.Conv2d(32, 64,  (2,4), 1, (1, 2))
		self.conv4_gate = nn.Conv2d(64, 64,  3,     1,  1)
		self.conv5      = nn.Conv2d(64, 128, 3,    1,  1)
		self.dropout    = nn.Dropout(p=0.3)

	def forward(self, x):
		x = self.conv1(x)
		x = self.tanh(x)
		x = self.pool(x)
		x = self.conv2(x)
		x = x * self.sigmoid(self.conv2_gate(x))
		x = self.tanh(x)
		x = self.pool(x)
		x = self.conv3(x)
		x = x * self.sigmoid(self.conv3_gate(x))
		x = self.tanh(x)
		x = self.pool(x)
		x = self.conv4(x)
		x = x * self.sigmoid(self.conv4_gate(x))
		x = self.tanh(x)
		x = self.pool(x)
		x = self.conv5(x)
		x = self.tanh(x)
		x = x.reshape(x.shape[0], x.shape[1], -1)
		x = self.dropout(x)
		return x 

class PositionalEncoding(nn.Module):
    "Implement the PE (exp) function for attention."
    def __init__(self, d_model, dropout, max_len=1681):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class AttentionInterface(nn.Module):
	"perform attention w positional encoding, \
	learns concept of left->right, top->bot reading."
	def __init__(self, dim):
		super(AttentionInterface, self).__init__()
		self.dim = dim
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, 
														nhead=2,
														dim_feedforward=256
														)
		self.posenc = PositionalEncoding(d_model=self.dim, dropout=0.2)
		self.enc = nn.TransformerEncoder(encoder_layer=self.encoder_layer, 
											num_layers=4, 
											norm=None
											)
	def forward(self, x):   
		x = self.posenc(x)
		x = self.enc(x)
		return x


class Decoder(nn.Module):
	"1D-bLSTM to predict characters, many to many."
	def __init__(self, hidden_size, num_layers, sequence_length):
		super(Decoder, self).__init__()
		self.tanh = nn.Tanh()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm1 = nn.LSTM(sequence_length, 
							hidden_size, 
							num_layers, 
							batch_first=True, 
							bidirectional=True,
							# dropout=0.25
							)
		self.fc1 = nn.Linear(hidden_size*2, hidden_size)
		self.lstm2 = nn.LSTM(hidden_size, 
							hidden_size, 
							num_layers, 
							batch_first=True, 
							bidirectional=True,
							# dropout=0.25 
							)
		self.fc2 = nn.Linear(hidden_size*2, alphabet)

	def forward(self, x):
		# Set initial hidden and cell states
		h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
		c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

		# fwd out: tensor of shape (batch_size, seq_length, hidden_size*2)
		out, state = self.lstm1(x, (h0, c0))
		out = self.fc1(out) # pass hidden size to fc layer
		out, state = self.lstm2(x, (h0, c0))
		out = self.fc2(out) # pass hidden size to fc layer
		return out

class Model(nn.Module):
	"Connects the Encoder and Decoder to produce the output."
	def __init__(self, targets=None):
		super(Model, self).__init__()
		self.targets = targets
		self.encoder = Encoder(in_channels=imsize[0], pool_height=imsize[1])
		self.attention_interface = AttentionInterface(dim=dim)
		self.decoder = Decoder(hidden_size=hidden_size, 
				  num_layers=num_layers,
				  sequence_length=hidden_size)
	def forward(self, x):
		x = self.encoder(x)
		x = torch.transpose(x, 2, 1)# (B, C, W) -> (B, W, C)
		x = self.attention_interface(x)
		x = self.decoder(x)
		x = x.permute(1, 0, 2) # for ctc loss
		return x 

if __name__ == "__main__":
	with torch.cuda.amp.autocast():
		with torch.no_grad():
			net = Model().to(device)
			y = net(torch.randn(batch_size, imsize[0], imsize[1], imsize[2]).to(device))
			print(y[0].size())
