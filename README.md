# HebHTR
Hebrew Handwritten Text Recognizer, based on Machine Learning. Implemented with TensorFlow and OpenCV.\
Model is based on [Harald Scheidl](https://github.com/githubharald) SimpleHTR model [1], and CTC-WordBeam algoritm [2].

<p align="center">
  <img width="40%" height="40%" src="https://user-images.githubusercontent.com/35609587/63640817-9f8d7300-c6ad-11e9-8f0d-56402f0fcf2d.png">
</p>


## Getting Started
### Prerequisites
Currently HebHTR is only supported on Linux. I've tested it on Ubuntu 18.04.

In order to run HebHTR you need to compile Harald Scheidl's CTC-WordBeam.
In order to do that you need to clone the [CTC-WordBeam](https://github.com/githubharald/CTCWordBeamSearch#usage),
go to cpp/proj/ directory and run the script ./buildTF.sh

### Quick Start
```python 
from HebHTR import *

# Create new HebHTR object.
img = HebHTR('example.png')

# Infer words from image.
text = img.imgToWords(iterations=5, decoder_type='word_beam', remove_vertical_lines=False,
                        remove_horziontal_lines=False)
```

Result:

<p align="center">
  <img width="80%" height="80%" src="https://user-images.githubusercontent.com/35609587/63640910-73262680-c6ae-11e9-936c-2f08c592def0.png">
</p>

## How This Works
HebHTR first detects the sentences in the text-based image. Then, for each sentence, it crops all the words in the sentence,
and passes each word to the model to decode the text written in it.

![1](https://user-images.githubusercontent.com/35609587/63639606-0acf4900-c69e-11e9-96e0-f66200d73993.png)

Word segmentation is based on one of my previous works which can be found [here](https://github.com/Lotemn102/TIS).

## About the Model
As mentioned, this model was written by Harald Scheidl. This model was trained to decode text from images with a single word.
I've trained the model on a Hebrew words dataset. The accuracy level of this model is 88%, with a character error rate around 4%.

The model receives input image of the shape 128*32, binary colored. It has 5 CNN layers, 2 RNN layers, and eventually words are being decoded with
a CTC-WordBeam algoritm. 

![2](https://user-images.githubusercontent.com/35609587/63640070-b11e4d00-c6a4-11e9-9034-06da6fb3d42e.png)

Explanation in much more details can be found in Harald's article [1].

All words prediced by this model should be fit it's input data, i.e binary colored images of size 128*32. Therefore, each image
is normalized to binary color. Then, it is resized (without distortion) until it either has a width of 128 or a height of 32.
Finally, it is copied into a (white) target image of size 128×32. 

The following figure demonstrates this process:

<p align="center">
  <img src="https://user-images.githubusercontent.com/35609587/63640655-72d85c00-c6ab-11e9-83a2-1837adda9546.png">
</p>

## About the Dataset
I've created a dataset of around 100,000 Hebrew words. Around 50,000 of them are real words, taken from students scanned exams.
Segementation of those words was done using one of my previous works which can be found [here](https://github.com/Lotemn102/TIS).\
This data was cleaned and labeled manually by me. The other 50,000 words were made artificially also by me. The word list for
creating the artificial words is taken from MILA's Hebrew stopwords lexicon [3].
Over all, the whole dataset contains 25 different handwritten fonts.
The dataset also contains digits and punctuation characters.

All words in the dataset have the size of 128×32, and were encoded into black and white (binary). \
For example:

<p align="center">
  <img src="https://user-images.githubusercontent.com/35609587/63640388-61418500-c6a8-11e9-87a4-5407723030ec.png">
</p>

## About the Corpus
The corpus which is being used in the Word Beam contains of around 500,000 unique Hebrew words.
The corpus was created by me using the MILA's Arutz 7 corpus [4], TheMarker corpus [5], HaKnesset corpus [6].

## Avaliable Functions
### imgToWords
```python
imgToWords(remove_horziontal_lines=False, remove_vertical_lines=False, iterations=5,
                    decoder_type='best_path')
```

Converts a text-based image to text. 

**Parameters:**

  - **remove_horziontal_lines** (bool): Whether to remove horizontal lines from the text or not.
     Default value is set to 'False'.
      
  - **remove_vertical_lines** (bool): Whether to remove vertical lines from the text or not.
     Default value is set to 'False'.

  - **iterations** (int): Number of dilation iterations that will be done on the image. Image is dilated to find
    the contours of it's words. efault value is set to 5.
    
  - **decoder_type** (string): Which decoder to use when infering a word. There are two decoding options:
     -  'word_beam' - CTC word beam algorithm.
     -  'best_path' - Determinded by taking the model's most likely character at each position.
     
     The word beam decoding has significant better results.
     
     
**Returns**
  - Text decoded by the model from the image (string).
---------


### drawRectangles
```python
drawRectangles(output_path=None, remove_horziontal_lines=False, remove_vertical_lines=False,
                        iterations=5, dilate=True)
```
This function draws rectangles around the words in the text. With this function, you can see how 'remove_horizontal_lines',
'remove_vertical_lines' and  'iterations' variables affect the HebHTR segmentation performance.

**Parameters:**
  - **output_path** (string): A path to save the image to.
       If None is given as a parameter, image will be saved in the original image parent directory.
       
  - **remove_horziontal_lines** (bool): Whether to remove horizontal lines from the text or not.
     Default value is set to 'False'.
      
  - **remove_vertical_lines** (bool): Whether to remove vertical lines from the text or not.
     Default value is set to 'False'.
     
  - **iterations** (int): Number of dilation iterations that will be done on the image. Image is dilated to find
    the contours of it's words. efault value is set to 5.
    
  - **dilate** (bool): Whether to dilate the text in the image or not. Default is set to 'True'.
       It is recommended to dilate the image for better segmentation.
       
 **Returns**
  - None. Saves the image in the output path. 
  
  
## Improve Accuracy
Model's accuracy is around 88%, but because of the word segmentation, for large texts accuracy might be much lower.\
I suggest two ways to improve it:

**1. Change number of iterations**. \
Higher number of iterations is suitable for large letters and a lot of spaces between words, while
   lower number of iterations is siutable for smaller handwrite. Use the **drawRectangles** function to see how the number of
   iterations affects HebHTR segmentation of your text.
   I will use the following sentence as an example:
   <p align="center">
   <img src="https://user-images.githubusercontent.com/35609587/63641024-3ce9a680-c6b0-11e9-851e-4107ffb524bb.png">
   </p>
   
 For **3** iterations we get the following segmentation:
  <p align="center">
   <img src="https://user-images.githubusercontent.com/35609587/63641038-6d314500-c6b0-11e9-8d10-8217e9460bcb.png">
   </p>
  
   Which the model infers as:
  <p align='center'>
  , כולת להקשום לעצמ נו - סוגיה מעני ות המשתנה עםהזמן
  </p>
  
  And for **6** iterations we get the following segmentation:
  <p align="center">
  <img src="https://user-images.githubusercontent.com/35609587/63641044-9520a880-c6b0-11e9-9dc6-9c078357c977.png">
  </p>
  Which the model infers as:
  <p align='center'>
  היכולת להקשיב לעמנו - סוגיה מעניינת המשתנה עם הזמן
  </p>
  
------

**2. Remove horizontal and/or vertical lines.** \
Removing those lines might improve sentences segmentation, and thus improve model's infering accuracy.

For example:

<p align="center">
<img src="https://user-images.githubusercontent.com/35609587/63641150-293f3f80-c6b2-11e9-9586-d46a5cd8a13c.png">
</p>

Without using any of the removing options, we get complete gibberish:
<p align='center'>
  4- א- תמ" - מו, רח או- ין אות הלחמה הברים+ מידווסט יות באלו ברוחם נ: ורם מוטי אות, מוטין, אל ליוי יורטי ודורי ידי מ- יוש: מלי. 
- ימש, - ואירופאים - צרפת - וסוריה ניתנה לצרפת. השושלת ההאשמית רצתה את השליטה בסוריה - בשם הלאום הערבי.
 </p>
 
 
 but when we use both of the removing options, we get:
 <p align='center'>
  והם כבשו את דמשק, אך לאחר המלחמה הבריטים העדיפו את בעלי בריתם האירופאים - צרפת - וסוריה ניתנה לצרפת. (השושלת ההאשמית רצתה את השליטה בסוריה - בשם הלאום הערבי.
 </p>
 
 ----

If none of the above helps, i suggest you try to do the word segmentation with another algorithm which fits to your data,
and then infer each word with the model.

 
## Requierments
  - TensorFlow 1.12.0 
  - Numpy 16.4 (will work on 17.0 as well)
  - OpenCV
    
## References
[1] [Harald Scheid's SimpleHTR model](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)\
[2] [Harald Scheid's CTC-WordBeam algorithm](https://towardsdatascience.com/word-beam-search-a-ctc-decoding-algorithm-b051d28f3d2e)\
[3] [The MILA Hebrew Lexicon](http://www.mila.cs.technion.ac.il/resources_lexicons_stopwords.html)\
[4] [MILA's Arutz 7 corpus](http://www.mila.cs.technion.ac.il/eng/resources_corpora_arutz7.html)\
[5] [MILA's TheMarker corpus](http://www.mila.cs.technion.ac.il/eng/resources_corpora_themarker.html)\
[6] [MILA's HaKnesset corpus](http://www.mila.cs.technion.ac.il/eng/resources_corpora_haknesset.html)

