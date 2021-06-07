'''
training.
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
import string
from sklearn import preprocessing
from sklearn.model_selection import KFold

from model import Model
from dataset import IAM_Dataset, collate_fn_padd


# Hyper-parameters
batch_size = 28
val_batch_size = 5
dim = hidden_size = 128
num_epochs = 25
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Img_path = r'C:\Users\F\Desktop\Snik\Datasets\IAM\formsA-D'
Data_path = r'C:\Users\F\Desktop\Snik\Datasets\IAM\ascii\words.txt'
val_path = r'C:\Users\F\Desktop\Snik\Datasets\IAM\formsE-H'


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print('done saving.')
def load_checkpoint(checkpoint):
	print('Loading checkpoint.')
	model.load_state_dict(checkpoint['state_dict'])


def train(save_model=False, load_model=False):
	if load_model:
		load_checkpoint(torch.load("my_checkpoint.pth.tar"))

	train_dataset = IAM_Dataset(img_path=Img_path, Data_path=Data_path, transform=None)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn_padd)
	val_dataset = IAM_Dataset(img_path=val_path, Data_path=Data_path, transform=None)
	val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn_padd)
	model = Model().to(device) 


	# scaler = torch.cuda.amp.GradScaler()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, factor=0.8, patience=5, verbose=True
		)

	for epoch in range(num_epochs):
		if save_model:
			if epoch % num_epochs-1 == 0:
				checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
				save_checkpoint(checkpoint)
		train_loss = train_fcn(train_loader, model, optimizer) #, scaler
		val = val_fcn(val_loader, model)
		scheduler.step(train_loss)
		print(total_loss, 'total training loss')
		print(val_loss, 'total validation loss')

def train_fcn(loader, model, optimizer): #, scaler

	loop = tqdm(loader)
	total_loss = 0
	for batch_idx, (data, targets) in enumerate(loop):

		'Collation, padding.'
		#Data. list of tensors --> tensor
		data_new = data[0]
		for i in range(1, len(data)):
			data_new = torch.cat((data[i], data_new), 0)
		data = data_new.unsqueeze(dim=1).to(device)

		#Targets. pad lengths AND list of tensors --> tensor
		targ_lens = [len(i.squeeze()) for i in targets]

		targets_new = torch.empty(max(targ_lens), 1)
		# Transpose flips the data order, so it must be reversed.
		for len_idx in reversed(range(len(targ_lens))):
			if max(targ_lens) > targ_lens[len_idx]:
				#pad to make all lengths the same
				padding = torch.full((max(targ_lens) - targ_lens[len_idx], 1), 72)
				targets[len_idx] = torch.cat((targets[len_idx], padding), 0)

				# list of tensors --> tensor. 
				targets_new = torch.cat((targets[len_idx], targets_new), 1) 
			else: # if this one is the max, just add it.
				targets_new = torch.cat((targets[len_idx], targets_new), 1) 

		# [letters, batch] -> [batch, letters]
		targets_new = targets_new.T[:batch_size]
		targets = targets_new.to(device) #adding channel dim

		#forward
		data_out = model(data)
		output = F.log_softmax(data_out, dim=2)
		input_lengths = torch.full(
				size=(batch_size,), fill_value=len(output[:,0,:]), dtype=torch.long
			)
		target_lengths = torch.full(
				size=(batch_size,), fill_value=len(targets[1]), dtype=torch.long
			)

		ctc_loss = nn.CTCLoss(blank=72, zero_infinity=True)
		loss = ctc_loss(output, targets, input_lengths, target_lengths)
		loss += total_loss

		#backwards
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loop.set_postfix(loss=loss.item())

	return total_loss

def val_fcn(loader, model):
	#to decode
	printable = []
	for i in string.printable:
	    printable.append(i)
	printable.append('blank') #72
	le = preprocessing.LabelEncoder()
	le.fit(printable)

	model.eval()
	loop = tqdm(loader)
	val_loss = 0
	for batch_idx, (data, targets) in enumerate(loop):

		'Collation, padding.'
		#Data. list of tensors --> tensor
		data_new = data[0]
		for i in range(1, len(data)):
			data_new = torch.cat((data[i], data_new), 0)
		data = data_new.unsqueeze(dim=1).to(device)

		#Targets. pad lengths AND list of tensors --> tensor
		targ_lens = [len(i.squeeze()) for i in targets]

		targets_new = torch.empty(max(targ_lens), 1)
		# Transpose flips the data order, so it must be reversed.
		for len_idx in reversed(range(len(targ_lens))):
			if max(targ_lens) > targ_lens[len_idx]:
				#pad to make all lengths the same
				padding = torch.full((max(targ_lens) - targ_lens[len_idx], 1), 72)
				targets[len_idx] = torch.cat((targets[len_idx], padding), 0)

				# list of tensors --> tensor. 
				targets_new = torch.cat((targets[len_idx], targets_new), 1) 
			else: # if this one is the max, just add it.
				targets_new = torch.cat((targets[len_idx], targets_new), 1) 

		# [letters, batch] -> [batch, letters]
		targets_new = targets_new.T[:val_batch_size]
		targets = targets_new.to(device) #adding channel dim

		#forward
		# with torch.cuda.amp.autocast():
		data_out = model(data)
		output = F.log_softmax(data_out, dim=2)
		input_lengths = torch.full(
				size=(val_batch_size,), fill_value=len(output[:,0,:]), dtype=torch.long
			)
		target_lengths = torch.full(
				size=(val_batch_size,), fill_value=len(targets[1]), dtype=torch.long
			)

		ctc_loss = nn.CTCLoss(blank=72, zero_infinity=True)
		loss = ctc_loss(output, targets, input_lengths, target_lengths)
		loss += val_loss

	return val_loss

if __name__ == "__main__":
	train()