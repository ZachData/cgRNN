# cgRNN
An implementation of the paper: "Gated Convolutional Recurrent Neural Networks for Multilingual Handwriting Recognition," by Theodore Bluche and Ronaldo Messina.

Find the paper here, it is quite well written:
http://www.tbluche.com/files/icdar17_gnn.pdf

We implement a Gated Convolutional network (gCNN) as an encoder attached to a bidirecitonal Recurrent Neural Network (bLSTM) as a decoder; CTC loss is then used in order to allow the network to learn representations of characters from images. This allows one to perform state-of-the-art handwritten character recognition. We train on the IAM dataset.

The 'gate' mechanism is a sigmoid over a kernel. This a a good prior if we assume that a kernel is well activated at the locations where the text is, and is less activated where text is not. The gating mechanism appears to be one of the most important parts of the network, paired with CTC loss. This is clearer if we read into the paper 'Pay Attention to What You Read,' which implements a Vision Transformer as an end-to-end solution to handwritten character recognition. 

The paper implements a custom attention mechanism between the CNN and RNN. A  We abandom this and implement a transformer encoder in its place. The positional encoding mechanism paired with attention should allow the network to easily organize the locations of interesting objects within some space. This translates into the model learning how to read from left to right, top to bottom. The Transformer model was chosen because of its incredible popularity and its beatifully simple implementation.

The RNN of the model performs basic NLP to make sure that the characters create a sensible string. If we wished to break from the model in the paper, a transformer decoder could have fit in just as well here.  
