# VR-head_movement-predictor
### About the Problem and Project
This project is part of a research project, where I am trying to increase the viability
of steaming 360 degree videos over a network.  These videos typically require 4 to 6 times the bandwidth
of a normal video, so additional steps need to be taken to try and cut down total data transmitted.

The way I aim to improve this is by streaming only the area in the video which is in the viewer's
current field of view (FOV).  This could save dramatically on transmission, because a person's field of view is around 120 degrees horizontally, and 55 degrees vertically; only a portion of the field of view within the sphere could be transmitted and still give a full experience.


<img align="center" height="300" src="https://qph.fs.quoracdn.net/main-qimg-a2766d8864f09f3d072cced721669f5f">
<img align="center" height="300" src="https://i.stack.imgur.com/f2Iza.jpg">

Constantly transmitting a user's field of view to and from a streaming server takes too much bandwidth in itself, so a predictive method would need to be used.  There has been a few people looking into predicting future field of view using heuristic solution, but this project is instead focused on using neural networks to learn how users move their heads while watching a 360 video.  If the network can be trained to predict frames up to 3 seconds in advance with relative accuracy, then that prediction can dictate what areas of the sphere are streamed in higher quality for a smooth experience.  The central idea is to give the network a sequence of previous head movements of where the user *has* looked, along with a saliency map corresponding to the gaze direction of each frame, and output a sequence of where the user *will* look given a the future saliency of the video.  

---
### Data

The data from this project was procured from [this](http://dash.ipv6.enstb.fr/headMovements/) study from 57 participants.  The page provides instruction on downloading the data, though it is included as part of this repository.  The page also provides links to download the videos, which are *not* part of the repository for space reasons.  

Before training examples can be loaded, the video being trained on must be processed for its saliency.  The way this is done is with `saliencyRecord.py`, which will look in the current directory for the video name, and output a HDF5 file representing an intensity map describing saliency of each frame of the video.  The saliency recorder downsizes the video for space reasons.  Output visualized would look like this:

<img align='center' height="300" src="/images/saliency_example.png">

`hmd_procure.py` will take care of sorting through the data in the folders `./vr_hmd_test` and `./vr_hmd_train`.  Import the module then create the data grabber with   
 `grabber = hmd_procure.HMDGrabber('train')` *(specify test or train set as parameter)*, and then load all the participants' data with `participants = grabber.grabData()`

 This will load the data into an array where each element is a Participant object, each containing VideoData object describing their movements for that video and some personal details like age and sex.  

 Using PyTorch's dataloader, `data.py` takes care of loading training examples.  To see the dataloader in action, `dataScript.py` shows an example of loading data and printing to console.  To create the dataloader, first the HDF5 saliency map must be created and loaded, then the paritipant data must be loaded, and then the dataloader can be created with `dataloader = ShortWindowDataLoader(hdFile, participants, frame_window_prev, frame_window_fut, bs, videoKey)`.  
 `frame_window_prev` and `frame_window_fut` describe the amount of previous and future frames you want for each training example, bs describes the batch size, and videoKey is the lowercase name of the video you are using, like: 'rollercoaster'

---

### Maths

Because this project has to do wtih 3D space and graphics to some degree, there is quite a bit of math that needs to be done to make sense of the projections and binning of the data.  `translate.py` contains useful functions for processing the data.  There are function to:
- Compute the Hamilton Product of two hamilton quaternions, 4-Dimensional imaginary numbers describing a rotation in 3D
- Convert polar coordinates to a 3D unit vector
- Rotate a 3D vector by a rotation described in a Hamilton Quaternion format
- Perform an equirectangular projection of a 3D vector onto an image described by `[w,h]`
- Return the BinID associated with a point on a 2D plane, given that the image is binned by `[row, col]`
- Return the coordinates of the lower left corner of a bin given its ID and the image dimensions
- Convert a sequence of head movement rotations from the datafile to a sequence of BinIDs


###### Note: many of these methods were implemented from scratch, and could possibly use vectorization for major speed improvements


---

### Understand The Data

With the data loading and maths being mentioned, these ideas are all brought together in the file `rectangview.py` where you can visualize the data from the study.  This file takes in a video name, dataset (test/train), binning specs, and resize parameter (specified in settings), and displays the video frame by frame in Matplotlib, with a grid representing the binning layout.  Each frame, the participants' viewing perspectives are calculated and translated into 3D vectors, then projected onto a 2D plane, then binned accordingly.  Each bin will show more video more clearly depending on how many people are currently viewing in that direction.  
<img align='center' height="300" src="/images/visualize.png">

---
### Neural Network Concepts

This project is attempting to use an encoder/decoder paradigm using LSTM cells with the fantastic PyTorch library.  Below is described the dataflow and features of the encoder/decoder architecture.  The definition of this model can be found in `model.py`
##### Encoder
   The encoder takes as input the Hamilton Quaternion representing the rotation of the user's head, the user's age, and the frame number for that recorder position, along with a saliency map for that frame of the video.  The saliency map is precomputed and stored in an HDF5 file using `saliencyRecord.py`, and is vectorized into a 1D vector *[TODO: could use convolution possibly?]*.  PyTorch takes as input to a LSTM cell a tensor of dimensions [input_seq_len x batch_size x data_size], so these features are concatenated accordingly to create the correct sized tensor.  

  As output, the encoder produces a 'context vector' which is then fed into the decoder.

  ##### Decoder
   The decoder tajes in the 'context vector' from the encoder, which is supposed to embody the meaning or representation of the sequence witnessed.  The decoder then sets this context vector as its hidden state, and feeds in a *StartOfSequence* vector to begin 'decoding' the context vector.  The output from each step of the sequence of decoding is a [1 x b] array representing the amount of bins used to separate the equi-rectangular projected video.

   ---

   ### Training

  The network can be trained with `train.py` , which contains many parameters at the beginnign of the file.  Not all of the parameters are working fully as of this posting, this is something to work on *(multiple LSTM layers have not been tested yet, only single cell architectures)*.  After training is complete, the models are saved in the current working directory with the names specified.  

  The file `model_eval.py` will load up the encoder and decoder models created, and allow qualitative judging of their accuracy.  Having specified the amount of frames desired to be predicted, the video name, binning scheme, and passes, the output will show what the network predicts the viewed bin will be for each frame, and what the viewed bin *actually was* for each frame.  

  ---

   ### Goals

   There is a lot to be done with this project.  Some of the most urgent items on the TODO list are mentioned below:
   1. Modify the model so that the decoder takes as input not only the *StartOfSequence* vector or previous output, but also the saliency map for the future frame.  This is a core idea as to how the network will have more information available to make predictions as to where the viewer will be looking, but has not been implemented yet.  
   2. Create a visualization show visually what bins the network predicts where the user will be looking instead of soley on console, visuals are always better
   3. Start to tune the network to output meaningful results.  Investigating different layers of LSTM cells, training hyperparemeters
   4. Possibly include convolution to the input saliency map to preserve some of the spacial dependencies.  
   5. Create a streaming simulation environment where the model is used to actively predict and stream certain areas with better/upgrading quality using model outputs.
   6. Code optimizations for faster training

---

### Additional Notes
- This project is large and right now is only worked on by me, but any help at all is eagerly welcomed.  
- Depending on your computing setup, some parts of the code will need to be modified for GPU or CPU usage during training.
- At first, I tried using a normal LSTM cell and a GRU cell to treat the sequence as a classification problem, but despite longer training times I decided an encoder/decoder paradigm would serve better for a sequence-to-sequence translation.  Thoughts on this overall architecture are very welcomed.  
- There is a known bug occasionally with loading some data and giving `NaN` as the projected point from 3D to 2D, this is something on the TODO list to fix, this is definitley just a math issue for some edge cases.  
