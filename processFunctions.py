import cv2
import numpy as np
import re

regex = r'^[.,/"=+_?!*%~\'{}\[\]:().,;]+$'

# Resize image to fit model's input size, and place it on model's size empty image.
def preprocessImageForPrediction(img, imgSize):
    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)),
                                                1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img

# Clean image background and sharpen text.
def sharpenText(img):
    # Check if image is already in gray scale.
    try:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        imgray = img

    # Blurring the image before and after threshold seems to have better
    # results.
    imgBlur = cv2.medianBlur(imgray, 3, 0)
    _, imgBin = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_OTSU)
    img = cv2.blur(imgBin, (1, 1))

    return img

# Find contours of words in the text.
def contours(img, iterations=3, dilate=True):
    if dilate is False:
        im = img
    else:
        # Dilate image for better segmentation
        im = dilateImage(img, iterations)

    # Check if image is already gray.
    try:
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    except:
        imgray = im

    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort contours by order of words in a sentence.
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Reverse sorted contours, for RTL.
    sorted_ctrs.reverse()

    return sorted_ctrs

# Find sentences boundaries in large text.
def findSentencesBoundaries(img, original_img):
    # Parts from this code are taken from:
    # https://stackoverflow.com/questions/34981144/split-text-lines-in
    # -scanned-document/35014061#35014061
    img = sharpenText(img)
    try:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        imgray = img

    h, w = imgray.shape

    ## (2) threshold
    th, threshed = cv2.threshold(imgray, 127, 255,
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    ## (3) minAreaRect on the nozeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx, cy), (W, H), ang = ret

    if W > H and h > w:
        ang = ang + 90

    if H > W and w > h:
        ang = ang + 90

    ## (4) Find rotated matrix, do rotation
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)

    th = 5
    H, W = img.shape[:2]
    lines = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]

    if not lines:
        return [img]

    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # Calculate average distance between every 2 lines, in order to clean very
    # close lines.
    lines_distances = []
    list_lines = []

    for i in range(0, len(lines)):
        dist = lines[i] - lines[i-1]
        if i is 0:
            lines_distances.append(0)
        # Remove very close lines.
        elif dist > 5:
            list_lines.append(lines[i])
            lines_distances.append(dist)

    final_lines = []

    if len(lines_distances) > 1:
        average_space = sum(lines_distances) / (len(lines_distances)-1)
    else:
        average_space = sum(lines_distances)

    lines_distances[0] = int(average_space)
    removed_last_one = False
    epsilon = 20 # Limit the allowed deviation from the average distance.
    shift = 15 # Move the lines a little bit down, for better segmentation.

    if not list_lines:
        return [img]

    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    imgray = sharpenText(img)

    for i in range(0, len(list_lines)+1):
        # For the first sentence.
        if i is 0:
            cv2.line(rotated, (0, list_lines[0] - shift), (W, list_lines[0] - shift), (0, 255, 0), 1)
            cropped = imgray[0:(list_lines[0] - shift), 0:W]
            if (cropped.shape[0] > 0) and (cropped.shape[1] > 0):
                final_lines.append(cropped)

        elif average_space + epsilon < lines_distances[i] or\
                average_space - epsilon > lines_distances[i]\
                and not removed_last_one:
            removed_last_one = True
        else:
            cv2.line(rotated, (0, list_lines[i-1] - shift),
                     (W, list_lines[i-1] - shift), (0, 255, 0), 1)
            cropped = imgray[(list_lines[i-2] - shift):(list_lines[i-1] - shift), 0:W]

            if (cropped.shape[0] > 0) and (cropped.shape[1] > 0):
                final_lines.append(cropped)

            removed_last_one = False

    # For the last sentence.
    cropped = imgray[(list_lines[len(list_lines) - 1] - shift):H, 0:W]
    if (cropped.shape[0] > 0) and (cropped.shape[1] > 0):
        final_lines.append(cropped)

    cv2.imwrite("result.png", rotated)
    return final_lines

# Dilate image for better segmentation in contours detection.
def dilateImage(img, iterations):
    # Convert image to gray.
    # Check if image is already in gray scale.
    try:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        imgray = img

    # Clean all noises.
    denoised = cv2.fastNlMeansDenoising(imgray, dst=None, h=10)

    # Negative the image.
    imagem = cv2.bitwise_not(denoised)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(imagem, kernel, iterations=iterations)

    # Negative it again to original color
    final = cv2.bitwise_not(dilate)

    return final

# Crop the words the contours function found.
def cropWords(img, conts):
    all_words = []

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        cropped = img[y:y + h, x:x + w]
        all_words.append(cropped)

    return all_words

def isPunctuation(some_str):
    search = re.compile(regex, re.UNICODE).search
    return bool(search(some_str))

def wordListToString(word_list):
    final_string = ''

    for word in word_list:
        # Words in hebrew can't start with 'ן'
        if word[0] == 'ן':
            temp_list = list(word)
            temp_list[0] = 'ו'
            word = ''.join(temp_list)

        # In Hebrew, you can't put spaces after punctuations.
        if isPunctuation(word):
            # Remove last space from the string.
            final_string = final_string[:-1]

        final_string += word
        final_string += ' '

        # For cases in which the word segmentation algorithm cropped some letter from a word.
        # For instance, it may crop the word "בנוסף" into "ב" and "נוסף".
        if len(word) is 1 and not isPunctuation(word) and word is not '-':
            final_string = final_string[:-1]

        if word == '.':
            final_string += '\n'

    if final_string[-1] == ' ':
        # Remove space at the end of the string.
        final_string = final_string[:-1]

    return final_string # Remove space at the end of the string.

# Draw rectangles around the contours function found.
def drawRects(img, contour, path_save):
    (x, y, w, h) = cv2.boundingRect(contour)

    # Clean all small contours out.
    if (h*w) < 25:
        return

    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imwrite(path_save, img)

# Removes vertical lines (e.g from notebook scans).
def removeVerticalLines(img):
    img = sharpenText(img)

     # Check if image is already in gray scale.
    try:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        imgray = img

    # Inverse the image.
    img_inverse = cv2.bitwise_not(imgray)

    # Change image to binary color.
    bw = cv2.adaptiveThreshold(img_inverse, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, -2)

    # Copy binary image.
    horizontal = bw.copy()

    # Find structuring element size. 30 is ok for most images i've tested in
    # various size from ~50*100 (a single word) to ~1500*2300 (full page).
    horiz_size = int(bw.shape[0] / (bw.shape[0]/30))

    # Create a line structuring element.
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (horiz_size, 1))

    # Preform erode function with structuring element in order to find the
    # image lines.
    horizontal = cv2.erode(src=horizontal, kernel=horizontal_structure,
                           anchor=(-1, -1))

    # Preform dilate function in order to connect some gaps in found lines.
    horizontal = cv2.dilate(src=horizontal, kernel=horizontal_structure,
                            anchor=(-1, -1))

    # Inverse the image, so that lines are black for masking.
    horizontal_inv = cv2.bitwise_not(horizontal)

    # Mask the inverted img with the inverted mask lines.
    masked_img = cv2.bitwise_and(img_inverse, img_inverse, mask=horizontal_inv)

    # Reverse the image back to normal.
    masked_img_inv = cv2.bitwise_not(masked_img)

    # Blur image and threshold it for better result.
    imgBlur = cv2.medianBlur(masked_img_inv, 3, 0)
    _, final = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_OTSU)

    return final

# Removes horizontal lines (e.g from notebook scans).
def removeHorizontalLines(img):
    img = sharpenText(img)

     # Check if image is already in gray scale.
    try:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        imgray = img

    # Inverse the image.
    img_inverse = cv2.bitwise_not(imgray)

    # Change image to binary color.
    bw = cv2.adaptiveThreshold(img_inverse, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, -2)

    # Copy binary image.
    horizontal = bw.copy()

    # Find structuring element size. 30 is ok for most images i've tested in
    # various size from ~50*100 (a single word) to ~1500*2300 (full page).
    horiz_size = int(bw.shape[1] / (bw.shape[1]/30))

    # Create a line structuring element.
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (horiz_size, 1))

    # Preform erode function with structuring element in order to find the
    # image lines.
    horizontal = cv2.erode(src=horizontal, kernel=horizontal_structure,
                           anchor=(-1, -1))

    # Preform dilate function in order to connect some gaps in found lines.
    horizontal = cv2.dilate(src=horizontal, kernel=horizontal_structure,
                            anchor=(-1, -1))

    # Inverse the image, so that lines are black for masking.
    horizontal_inv = cv2.bitwise_not(horizontal)

    # Mask the inverted img with the inverted mask lines.
    masked_img = cv2.bitwise_and(img_inverse, img_inverse, mask=horizontal_inv)

    # Reverse the image back to normal.
    masked_img_inv = cv2.bitwise_not(masked_img)

    # Blur image and threshold it for better result.
    imgBlur = cv2.medianBlur(masked_img_inv, 3, 0)
    _, final = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_OTSU)

    return final















