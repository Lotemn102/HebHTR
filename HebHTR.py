from processFunctions import *
from predictWord import *
import os

class HebHTR:

    def __init__(self, img_path):
        self.img_path = img_path
        self.original_img = cv2.imread(img_path)


    def imgToWord(self, iterations=5, decoder_type='best_path'):
        transcribed_words = []
        model = getModel(decoder_type=decoder_type)
        transcribed_words.extend(predictWord(self.original_img, model))
        final = wordListToString(transcribed_words)
        return final


    def drawRectangles(self, output_path=None, remove_horziontal_lines=False, remove_vertical_lines=False,
                        iterations=5, dilate=True):

        if remove_horziontal_lines:
            clean_h = removeHorizontalLines(self.original_img)
        else:
            clean_h = self.original_img

        if remove_vertical_lines:
            clean_v = removeVerticalLines(clean_h)
        else:
            clean_v = clean_h

        sharp_img = sharpenText(clean_v)
        conts = contours(sharp_img, iterations, dilate)

        if not output_path:
            parent_path = os.path.dirname(self.img_path)
            output_path = parent_path + '/draw_rectangles_result.png'

        for i in conts:
            drawRects(sharp_img, i, str(output_path))
