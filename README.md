# Deepfake-Detection
## Introdution
Deepfake is a technique to generate realistic videos and photos of fake events with a high potential
to deceive, which may mislead viewers’ information consumption. It is a source of misinformation,
intimidation, harassment, and manipulation when being malevolently used. Therefore, how to detect
deepfakes has become a serious problem. To solve this problem, some tech companies and academics
cooperate to build the Deepfake Detection Challenge (DFDC) on Kaggle, and we leverage
deep learning models, e.g. CNN, RNN, and attention, to detect these manipulated videos.

## Methodologies
### Preprocessing
For each video,  10 frames extracted uniformly along the time and the label is saved as one training data. As deepfake technology is used to generate artificial faces to replace the original faces in videos, what needs to be focused on is only the face part. Therefore, we applied face detection (MTCNN) to capture faces in all frames first. Then, in order to fit the input size of pretrained convolutional neural networks, all images are resized to either 224\*224 for VGG16 and ResNet-50, or 299\*299 for Inception V3. Data normalization is also applied to normalize three channels of images.

### CNN Model
In convolutional neural network (CNN) models, some popular frameworks, such as VGG16, ResNet-50, and Inception V3, are used to classify whether the video is fake or not. Each frame as an input goes through a CNN network and outputs a value. Then, one linear layer is added on the top of the structure to ensemble outputs from ten frames and output a final score to determine whether the video is fake. The CNN networks can be separated into two parts, convolutional layers and classifier layers. At the beginning of the network, the image would be passed through multiple convolutional layers. These layers extract latent image features, which is the critical process to detect anything abnormal in the images. After convolutional layers, the data would pass through one to three linear layers to classify the image. With the last layer added on the top, the output of ten frames would be considered to make the final decision for the video.
<p align="center">
  <img src="https://lh4.googleusercontent.com/HMKmztJ9Vl6WXvtjvDCvdqYYeJhW54OiWIul-xzzMEly1S6BBKGk1ODtg3jvRhggpNYIZBCZRPLWEclpYFQ6n05JWmiJsWm9q07mW64SQW_HkF1ehgLmdFyTYcwKD81Q3r9ZnT04" alt="img" width="350"/>
</p>

### RNN Model
As there are abnormal changes on the face when the person is talking if the videos generated from deepfake, we also apply recurrent neural network (RNN) to leverage temporal features for detection. In RNN models, each frame would pass through a ResNet-50 network without the last classifier layer to extract features and transform frames to embedding vectors. Then, ten embedding vectors would be regarded as a sequence and pass to the recurrent neural network. Here we use long-term short memory (LSTM) cells for RNN network. After the RNN network, two approaches are used in our experiments. One is taking the last hidden state as the vector containing all information of the sequence and passing through a linear layer to receive the final output probability. The other approach is using max pooling among all hidden states and doing the linear transformation to receive the final result.

<p align="center">
  <img src="https://i.imgur.com/4a0l1MV.png" alt="img"  width="620"/>
</p>

### Attention Model
Another method that can address temporal features is attention. It reads all input images in different timestamps and figures out which inputs should attend more when predicting for one timestamp. In our models, self-attention is applied to the embedding feature vectors for frames. The concept is that during the self-attention, the model match frames from different timestamp and attend some frames to compare their differences. After the self-attention layer, either concatenating vectors for all timestamps or applying max-pooling are conducted in our experiments. On top of the model, a linear layer is added to predict the output probability.

<p align="center">
  <img src="https://lh4.googleusercontent.com/G9mXGFQIPnxHzQgDxK1K4KzWO2puLB2lmFwglzgfcYT9fTdpjPfmSrO1qqZrKVPIC9SIN7lHJ_vNDxYuvtCoBgLqXFpcFLUs8scf_er3JKyxx4dWmjnUJK6Nt_qUP5SsoNCHq92F" alt="img"  width="400"/>
 </p>
