# Hebrew Handwritten Text Recognizer (OCR)
Hebrew Handwritten Text Recognizer, based on machine learning. Implemented with TensorFlow and OpenCV.\
Model is based on [Harald Scheidl](https://github.com/githubharald)'s SimpleHTR model [1], and CTC-WordBeam algorithm [2].

<p align="center">
  <img width="40%" height="40%" src="https://user-images.githubusercontent.com/35609587/63640817-9f8d7300-c6ad-11e9-8f0d-56402f0fcf2d.png">
</p>


## Getting Started
### Prerequisites
Currently HebHTR is only supported on Linux. I've tested it on Ubuntu 18.04.

In order to run HebHTR you need to compile Harald Scheidl's CTC-WordBeam.
In order to do that you need to clone the [CTC-WordBeam](https://github.com/githubharald/CTCWordBeamSearch#usage),
go to ```cpp/proj/ directory``` and run the script ```./buildTF.sh```.

### Quick Start
```python 
from HebHTR import *

# Create new HebHTR object.
img = HebHTR('example.png')

# Infer words from image.
text = img.imgToWord(iterations=5, decoder_type='word_beam')
```

Result:

<p align="center">
  <img width="80%" height="80%" src="https://user-images.githubusercontent.com/35609587/63640910-73262680-c6ae-11e9-936c-2f08c592def0.png">
</p>

## About the Model
As mentioned, this model was written by Harald Scheidl. This model was trained to decode text from images with a single word.
I've trained the model on a Hebrew words dataset.

The model receives input image of the shape 128×32, binary colored. It has 5 CNN layers, 2 RNN layers, and eventually words are being decoded with
a CTC-WordBeam algoritm. 

![2](https://user-images.githubusercontent.com/35609587/63640070-b11e4d00-c6a4-11e9-9034-06da6fb3d42e.png)

Explanation in much more details can be found in Harald's article [1].

All words prediced by this model should fit it's input data, i.e binary colored images of size 128*32. Therefore, HebHTR
normalizes each image to binary color. Then, HebHTR resizes it (without distortion) until it either has a width of 128 or a height of 32. Finally, image is copied into a (white) target image of size 128×32. 

The following figure demonstrates this process:

<p align="center">
  <img src="https://user-images.githubusercontent.com/35609587/63646959-49f3ad80-c723-11e9-9c81-e41e2c8f5af7.png">
</p>

## About the Dataset
I've created a dataset of around 100,000 Hebrew words. Around 50,000 of them are real words, taken from students scanned exams.
Segementation of those words was done using one of my previous works which can be found [here](https://github.com/Lotemn102/TIS).\
This data was cleaned and labeled manually by me. The other 50,000 words were made artificially also by me. The word list for
creating the artificial words is taken from MILA's Hebrew stopwords lexicon [3].
Overall, the whole dataset contains 25 different handwrites.
The dataset also contains digits and punctuation characters.

All words in the dataset were encoded into black and white (binary). \
For example:

<p align="center">
  <img src="https://user-images.githubusercontent.com/35609587/63640388-61418500-c6a8-11e9-87a4-5407723030ec.png">
</p>

## About the Corpus
The corpus which is being used in the Word Beam contains around 500,000 unique Hebrew words.
The corpus was created by me using the MILA's Arutz 7 corpus [4], TheMarker corpus [5] and HaKnesset corpus [6].

## Avaliable Functions
### imgToWords
```python
imgToWords(iterations=5, decoder_type='word_beam')
```

Converts a text-based image to text. 

**Parameters:**

  - **iterations** (int): Number of dilation iterations that will be done on the image. Image is dilated to find
    the contours of it's words. Default value is set to 5.
    
  - **decoder_type** (string): Which decoder to use when infering a word. There are two decoding options:
     -  'word_beam' - CTC word beam algorithm.
     -  'best_path' - Determinded by taking the model's most likely character at each position.
     
     The word beam decoding has significant better results.
     
     
**Returns**
  - Text decoded by the model from the image (string).  

**Example of usage in this function:**
```python 
from HebHTR import *

# Create new HebHTR object.
img = HebHTR('example.png')

# Infer words from image.
text = img.imgToWord(iterations=5, decoder_type='word_beam')
```

Result:

<p align="center">
  <img width="80%" height="80%" src="https://user-images.githubusercontent.com/35609587/63640910-73262680-c6ae-11e9-936c-2f08c592def0.png">
</p>

---------
 
## Requirements
  - TensorFlow 1.12.0 
  - Numpy 16.4
  - OpenCV
    
## References
[1] [Harald Scheid's SimpleHTR model](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)\
[2] [Harald Scheid's CTC-WordBeam algorithm](https://towardsdatascience.com/word-beam-search-a-ctc-decoding-algorithm-b051d28f3d2e)\
[3] [The MILA Hebrew Lexicon](http://www.mila.cs.technion.ac.il/resources_lexicons_stopwords.html)\
[4] [MILA's Arutz 7 corpus](http://www.mila.cs.technion.ac.il/eng/resources_corpora_arutz7.html)\
[5] [MILA's TheMarker corpus](http://www.mila.cs.technion.ac.il/eng/resources_corpora_themarker.html)\
[6] [MILA's HaKnesset corpus](http://www.mila.cs.technion.ac.il/eng/resources_corpora_haknesset.html)

