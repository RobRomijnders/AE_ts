## Auto encoder for time series
This repo presents a simple auto encoder for time series. It visualizes the embeddings using both PCA and tSNE. I show this on a dataset of 5000 ECG's. The model doesn't use
the labels during training. Yet, the produced clusters visually separate the classes of ECG's.

People repeatedly ask me how to find patterns in time series using ML. The usual wavelet transforms and other features fail to yield results. They wonder
what ML has to offer. 

  * For categorical data, a usual choice are techniques like PCA, tSNE. 
  * For images, a usual choice is convolutional auto encoders
  * For time series, what is the usual choice?
    * This repo implements a recurrent auto encoder

## Why use a Recurrent Neural Network in an auto encoder?

  * The length of time series may vary from sample to sample. Conventional techniques only work on inputs of fixed size.
  * The patterns in time series can have arbitrary time span and be non stationary. The recurrent neural network can learn patterns in arbitrary time scaling.
The convolutional net, however, assumes only stationary patterns

## The network
_From here on, RNN refers to our Recurrent Neural Network architecture, the Long Short-term memory_
Our network in *AE_ts_model.py* has four main blocks

  * The **encoder** is a RNN that takes a sequence of input vectors
  * The **encoder to latent vector** is a linear layer that maps the final hidden vector of the RNN to a latent vector
  * The **latent vector to decoder** is a linear layer that maps the latent vector to the input vector for the decoder
  * The **decoder** is a RNN that takes this single input vector and maps to a sequence of output vectors

## Training Objective
An auto encoder learns the identity function, so the sequence of input and output vectors must be similar. In our case, we take a probabilistic approach.
Every output is a tuple of a mean, mu and standard deviation. Let this mu and sigma parametrize a Gaussian distribution. Now we minimize the log-likelihood
of the input under this distribution. We train this using backpropagation into the weights of the encoder, decoder and linear layers.

## Example data
I showcase the recurrent auto encoder on a dataset of 5000 ECG's. Accurately named ECG5000 on the [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/). I choose
ECG, because humans understand them easily. Yet, their complexity remains challenging enough for a machine learning model.

Here are some examples:
![examples](https://github.com/RobRomijnders/AE_ts/blob/master/im/data_examples.png?raw=true)
