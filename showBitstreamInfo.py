import random
import os
import functions
import shlex, subprocess
import numpy as np
import time
import cv2
import functions
import struct
import math
import sys


# A function to get the QP for each MB from a bunch of video files and histogram it (or something)

saveFolder = "qpFiles/"

datadir = '/Volumes/LaCie/data/yuv'
#videoFilesBase = 'Data/VID/snippets/train/ILSVRC2017_VID_train_0000/'
#annotFilesBase = 'Annotations/VID/train/ILSVRC2017_VID_train_0000/'
#baseFileName = 'ILSVRC2017_train_00000000'

x264 = "x264"
ldecod = "ldecod"
ldecod2 = "ldecod2"
ffmpeg = "ffmpeg"

from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

fileSizes = [
    ['qcif', 176, 144],
    ['512x384', 512, 384],
    ['384x512', 384, 512],
    ['cif', 352, 288],
    ['sif', 352, 240],
    ['720p', 1280, 720],
    ['1080p', 1920, 1080]
]

quants = [0, 7, 14, 21, 28, 35, 42, 49]

def createFileList(myDir, takeAll = False, format='.yuv', desiredNamePart = '_cif', shuffle=True):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(myDir):
        for filename in filenames:
            if filename.endswith(format):
                if takeAll or desiredNamePart in filename:
                    fileName = os.path.join(myDir, dirName, filename)
                    baseFileName, ext = os.path.splitext(filename)
                    print("The filename is {} baseFileName {}".format(fileName, baseFileName))

                    if format=='.yuv':
                        for fileSize in fileSizes:
                            if fileSize[0] in fileName:
                                tuple = [fileName, fileSize[1], fileSize[2]]
                                fileList.append(tuple)
                                break
                    elif format=='.tif':
                        tuple = [fileName, -1, -1]
                        fileList.append(tuple)

                    else:
                        fileList.append(fileName)

    #hacky hack
    #fileList = [['/Volumes/LaCie/data/yuv_quant_noDeblock/quant_0/mobile_cif_q0.yuv', 352, 288],]
    if shuffle:
        random.shuffle(fileList)
    #print(fileList)
    return fileList

def processMyldecodOP(filename, repeatSkipped=False):
    intras = []
    skippeds = []
    mvs = []
    qps = []
    frameCount = 0
    maxMB = 0
    frameData = []
    width = 1280
    height = 720


    with open(filename, "r") as f:
        frameMax = -1
        lastKeyFrame = -1
        currentFrame = -1
        for line in f:
            #print(line)
            line.strip()
            line = line.replace("\n", "")
            line = line.split(';')
            #print(line[0])
            if "Dimensions" in line[0]:
                terms = line[0].split()
                width = terms[1]
                height = terms[2]
                continue
            if "(I)" in line:
                print("Intra Frame {}".format(frameCount))
            if "(P)" in line:
                print("Intra Frame {}".format(frameCount))
            if "(B)" in line:
                print("Intra Frame {}".format(frameCount))
            if "Frame" not in line[0]:
                continue
            if len(line) <= 1:
                continue
            if "MB no:" not in line[1]:
                continue

            #print(line)
            intraterms = 1
            skippedterms = 0
            mvterms = []
            qpterms = 0
            frameterms = 0
            mbnoterms = 0
            for terms in line:
                terms = terms.strip()
                terms = terms.split(':')
                #print("First term: {} value: ".format(terms[0]))
                if terms[0] == "qp":
                    qpterms = int(terms[1])
                    qps.append(terms[1])
                elif terms[0] == "intra":
                    intraterms = int(terms[1])
                    intras.append(terms[1])
                elif terms[0] == "skipped":
                    skippedterms = int(terms[1])
                    skippeds.append(terms[1])
                elif terms[0] == "Motion vectors":
                    s = terms[1].replace(') (', ', ')
                    s = s.replace('(','')
                    s = s.replace(')','')
                    s = s.strip()
                    s = s.split(',')
                    mvterms = [int(a) for a in s]
                    mvs.append(s)
                elif terms[0] == "Frame":
                    frameterms = int(terms[1])
                    currentFrame = int(terms[1])
                    if frameterms == 0:
                        lastKeyFrame = frameMax + 1
                    if frameterms > frameMax:
                        frameMax = frameterms
                elif terms[0] == "MB no":
                    mbnoterms = int(terms[1])
                    #print("This is macroblock number {}".format(mbnoterms))
                    if mbnoterms == 0:
                        frameCount = lastKeyFrame + currentFrame
                        #print("{} = {} + {}".format(frameCount, lastKeyFrame, currentFrame))

                    if mbnoterms > maxMB:
                        maxMB = mbnoterms

            myTuple = [frameCount, mbnoterms, intraterms, skippedterms, qpterms]
            for mv in mvterms:
                myTuple.append(mv)
            tupleLen = len(myTuple)
            frameData.append(myTuple)
            #print(myTuple)

    #now turn frameData into a proper array.
    flat_list = [item for sublist in frameData for item in sublist]

    #print(mvs)

    frameCount = frameCount + 1 # to account for frame zero
    print("Number of frames from frameCount = {}".format(frameCount))
    frameData = np.asarray(flat_list)

    numFrames = (frameData.shape[0]/tupleLen)/maxMB
    print("Number of frames from frameCount = {}".format(numFrames))
    frameData = np.reshape(frameData, (frameData.shape[0]/tupleLen, tupleLen))

    #if there are "skipped" macroblocks, repeat the mode for the
    if repeatSkipped:
        print("We're supposed to do something here")

    # reorder/sort to account for B-frames
    frameData = frameData[frameData[:, 0].argsort(kind='mergesort')]
    #for item in frameData:
    #    print(item)
    print("The frame data shape")
    print(frameData.shape)
    print("Dimensions: {} {}".format(width, height))

    return frameData, int(width), int(height)

def demuxFromMPEG4toH264AnnexB(inFilename, outFilename):
    baseFileName, ext = os.path.splitext(inFilename)
    print("The filename is {} baseFileName {}".format(inFilename, baseFileName))

    # First demux mp4 to AnnexB (.264)
    try:
        os.remove(outFilename)
    except Exception as e:
        print("The file wasn't there?: {}".format(e))
    app = ffmpeg
    appargs = "-i {} -codec copy -bsf:v h264_mp4toannexb {}".format(inFilename, outFilename)
    exe = app + " " + appargs
    args = shlex.split(exe)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # print("Started subproc for ffmpeg")
    proc.wait()
    # print("Finished subproc for ffmpeg")
    # out, err = proc.communicate()

def processAfile(filename, width=1280, height=720, noRedos = True):
    baseFileName, ext = os.path.splitext(filename)
    traceFileName = "{}_trace.txt".format(baseFileName)
    print("The extension is {}".format(ext))
    h264filename = "{}.h264".format(baseFileName)
    if ("h264" not in ext):
        print("Use ffmpeg to convert")
        demuxFromMPEG4toH264AnnexB(filename, h264filename)
    if os.path.isfile(traceFileName) and noRedos:
        print("Trace file already exists")
    else:
        print("Creating the trace file")
        #ldecod -p InputFile = ().h264 > {}_op.txt
        useJM19 = True
        yuvFileName = "{}.yuv".format(baseFileName)
        if useJM19:
            app = ldecod
            appargs = "-p InputFile = {}.h264 -p OutputFile={}.yuv".format(baseFileName, baseFileName)
        else:
            # Using JM 7.5 or something - an old version before the standard was finalised
            app = ldecod2
            ldecodConfigName = "{}.cfg".format(baseFileName)
            prepareLdecodConfigFile(ldecodConfigName, h264filename, yuvFileName)
            appargs = ldecodConfigName
        exe = app + " " + appargs
        args = shlex.split(exe)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #print("Started subproc for ldecod")
        #proc.wait()
        #print("Finished subproc for ldecod")
        out, err = proc.communicate()
        with open(traceFileName, 'w') as f:
            f.write(out)

    return processMyldecodOP(traceFileName)

def prepareLdecodConfigFile(cfgName, inName, outName):
    with open(cfgName, "w") as f:
        f.write("{} \n".format(inName))
        f.write("{} \n".format(outName))
        f.write("test_rec.yuv\n")
        f.write("10\n")
        f.write("0\n")
        f.write("500000\n")
        f.write("104000\n")
        f.write("73000\n")
        f.write("\Users\pam\Documents\dev\JM_75br1\bin\leakybucketparam.cfg\n")

def getBGRFrameFromYUV420File(filename, width, height, frameNumber, RGB=False):
    yuvFrameSize = int(width * height * 3 / 2)
    bytePos = yuvFrameSize * frameNumber
    offEnd = 0
    with open(filename, "rb") as f:
        #pixels = []
        #for i in range(0, yuvFrameSize):
        #    pixels = f.read(yuvFrameSize)
        #    pixel = struct.unpack('c', p)
        #    pixels.append(pixels)
        #print(pixels)
        #print(bytePos)
        f.seek(bytePos)
        pixels = f.read(yuvFrameSize)
        pixels = bytearray(pixels)

        #print("Obtained len {} vs {}".format(len(pixels), yuvFrameSize))

        if len(pixels) != yuvFrameSize:
            #wrap round when we get to the end...
            f.seek(0)
            pixels = f.read(yuvFrameSize)
            pixels = bytearray(pixels)
            offEnd = 1


        pixels = np.asarray(pixels)
        #print("The pixels size: {}".format(pixels.shape))
        yuv444 = functions.YUV420_2_YUV444(pixels, height, width)
        if RGB:
            pixels = functions.planarYUV_2_planarRGB(yuv444, height, width)
        else:
            pixels = functions.planarYUV_2_planarBGR(yuv444, height, width)
        pixels = pixels.reshape((3, height, width))
        pixels = np.swapaxes(pixels,0,1)
        pixels = np.swapaxes(pixels,1,2)
    return offEnd, pixels

def getBGRFrameFromYUVOneChannelFile(filename, width, height, frameNumber, RGB=False, channels=1):
    #print("Channels: {} width: {} height: {}".format(channels, width, height))
    yuvFrameSize = int(width * height * channels)
    bytePos = yuvFrameSize * frameNumber
    offEnd = 0
    with open(filename, "rb") as f:
        #pixels = []
        #for i in range(0, yuvFrameSize):
        #    pixels = f.read(yuvFrameSize)
        #    pixel = struct.unpack('c', p)
        #    pixels.append(pixels)
        #print(pixels)
        #print(bytePos)
        f.seek(bytePos)
        pixels = f.read(yuvFrameSize)
        pixels = bytearray(pixels)

        #print("Obtained len {} vs {}".format(len(pixels), yuvFrameSize))

        if len(pixels) != yuvFrameSize:
            #wrap round when we get to the end...
            print('Wrap')
            f.seek(0)
            pixels = f.read(yuvFrameSize)
            pixels = bytearray(pixels)
            offEnd = 1


        pixels = np.asarray(pixels)
        #print("The pixels size: {}".format(pixels.shape))
        uvSize = (width * height * (3-channels))
        uvValues = np.full(uvSize, 128)
        if channels > 1:
            print("The frame before append:{}".format(pixels))
            print("The append:{}".format(uvValues))
        yuv444 = np.append(pixels, uvValues)
        if channels > 1:
            print("YUV bytes: {}".format(yuvFrameSize))
            print("bytepos: {}".format(bytePos))
            print("The frame:{}".format(yuv444))

        if RGB:
            pixels = functions.planarYUV_2_planarRGB(yuv444, height, width)
        else:
            pixels = functions.planarYUV_2_planarBGR(yuv444, height, width)
        pixels = pixels.reshape((3, height, width))
        pixels = np.swapaxes(pixels,0,1)
        pixels = np.swapaxes(pixels,1,2)
    return offEnd, pixels

def getBGRFramesFromMVsFile(filename, width, height, frameNumber, RGB=False, channels=2):
    # returns two frames, intensity means mv size, colour means direction
    #print("Channels: {} width: {} height: {}".format(channels, width, height))
    yuvFrameSize = int(width * height * channels)
    bytePos = yuvFrameSize * frameNumber
    offEnd = 0
    with open(filename, "rb") as f:
        #pixels = []
        #for i in range(0, yuvFrameSize):
        #    pixels = f.read(yuvFrameSize)
        #    pixel = struct.unpack('c', p)
        #    pixels.append(pixels)
        #print(pixels)
        #print(bytePos)
        f.seek(bytePos)
        pixels = f.read(yuvFrameSize)
        pixels = bytearray(pixels)

        #print("Obtained len {} vs {}".format(len(pixels), yuvFrameSize))

        if len(pixels) != yuvFrameSize:
            #wrap round when we get to the end...
            print('Wrap')
            f.seek(0)
            pixels = f.read(yuvFrameSize)
            pixels = bytearray(pixels)
            offEnd = 1


        pixels = np.asarray(pixels)
        xys = int(pixels.shape[0] / 2)
        pixels = np.reshape(pixels, (2,xys))
        pixelsX = pixels[0,:]
        pixelsY = pixels[1,:]
        pixelsX = pixelsX.flatten()
        pixelsY = pixelsY.flatten()



        #print("The pixels size: {}".format(pixels.shape))
        uvSize = (width * height)
        uvValuesX = np.full((2, uvSize), 128)
        uvValuesY = np.full((2, uvSize), 128)
        for idx, x in enumerate(pixelsX):
            pixelsX[idx] = abs(x)
            if x < 128:
                uvValuesX[0][idx] = 255
                uvValuesX[1][idx] = 0
            elif x > 128:
                uvValuesX[0][idx] = 0
                uvValuesX[1][idx] = 255
        uvValuesX = uvValuesX.flatten()
        for idx, y in enumerate(pixelsY):
            pixelsY[idx] = abs(y)
            if y < 128:
                uvValuesY[0][idx] = 255
                uvValuesY[1][idx] = 0
            elif y > 128:
                uvValuesY[0][idx] = 0
                uvValuesY[1][idx] = 255
        uvValuesY = uvValuesY.flatten()

        yuv444X = np.append(pixelsX, uvValuesX)
        yuv444Y = np.append(pixelsY, uvValuesY)

        if RGB:
            pixelsX = functions.planarYUV_2_planarRGB(yuv444X, height, width)
            pixelsY = functions.planarYUV_2_planarRGB(yuv444Y, height, width)
        else:
            pixelsX = functions.planarYUV_2_planarBGR(yuv444X, height, width)
            pixelsY = functions.planarYUV_2_planarBGR(yuv444Y, height, width)

        pixelsX = pixelsX.reshape((3, height, width))
        pixelsX = np.swapaxes(pixelsX,0,1)
        pixelsX = np.swapaxes(pixelsX,1,2)
        pixelsY = pixelsY.reshape((3, height, width))
        pixelsY = np.swapaxes(pixelsY, 0, 1)
        pixelsY = np.swapaxes(pixelsY, 1, 2)
    return offEnd, pixelsX, pixelsY

def turnElementsIntoFrame(frameData, frameNo, width, height, element):
    mbWidth = int((width+8)/16)
    mbHeight = int((height+8)/16)
    maxMbs = mbWidth * mbHeight

    bVal = 128
    gVal = 128

    start = maxMbs * frameNo
    end = maxMbs * (frameNo+1)
    #[frameCount, mbnoterms, intraterms, skippedterms, qpterms]
    if "qp" in element:
        elements = frameData[start:end, 4]
    elif "intra" in element:
        elements = frameData[start:end, 2]
    elif "skip" in element:
        elements = frameData[start:end, 3]
    elif "mv" in element:
        mvs = frameData[start:end, 5:]
        #TODO! Sort out the motion vectors. X and Y probably
        elements = frameData[start:end, 5]

    elements = np.asarray(elements).reshape(mbHeight, mbWidth)
    # Now to resize:
    bigElements = np.zeros((height, width, 3))
    for j in range(0,height):
        for i in range(0,width):
            #idash = int(math.floor((i+8)/16))
            #jdash = int(math.floor((j+8)/16))
            idash = int(math.floor(i/16))
            jdash = int(math.floor(j/16))
            #print("dash: ({},{}) from ({}, {})".format(idash, jdash, i, j))
            bigElements[j,i,0] = elements[jdash, idash]
            bigElements[j,i,1] = gVal
            bigElements[j,i,2] = bVal

    return bigElements

def scaleToVisualise(pixels, origW, origH, displayW, displayH, visible = True, uMax = 255, uMin = 0):
    if uMin == uMax:
        maximum = np.max(pixels)
        minimum = np.min(pixels)
    else:
        maximum = uMax
        minimum = uMin
    # make it visible
    if visible:
        scaleFactor = int(255/(maximum-minimum))
        pixels = pixels*scaleFactor

    # rescale to image dimensions
    dst = cv2.resize(pixels, (displayW, displayH), interpolation=cv2.INTER_NEAREST)
    #print(pixels)

    return dst





if __name__ == "__main__":
    loop = True
    play = True
    frameNo = 0
    increment = 1
    showMetaData = False
    showFourier = False
    showImage = True

    #options
    showFrameNo = False
    showMbNo = False
    showMbMode = False
    showQP = False
    showMV = False
    showRes = False
    showDiff = True

    #Display a file
    filename = "/Users/pam/Documents/data/DeepFakes/creepy2.yuv"
    filename = "/Users/pam/Documents/data/h264/carphone_qcif_q0.yuv"
    jpegFilename = "/Users/pam/Documents/data/pictures/capybara.jpg"
    filename = "/Users/pam/Documents/data/DeepFakes/creepy1.yuv"
    height = 720
    width = 1280
    #height = 144
    #width = 176
    maxDisplayHeight = 480
    maxDisplayWidth = 640
    doResize = False

    mbwidth = int(width/16)
    mbheight = int(height/16)

    basefilename, ext = os.path.splitext(filename)
    YUVfilename = "{}.yuv".format(basefilename)
    FrameNofilename = "{}.frameno".format(basefilename)
    MbNofilename = "{}.mbno".format(basefilename)
    MBModefilename = "{}.mbmode".format(basefilename)
    QPfilename = "{}.qp".format(basefilename)
    MVfilename = "{}.mv".format(basefilename)



    frameData = 0
    #filename = "/Users/pam/Documents/data/DeepFakes/creepy2.mp4"
    if showMetaData:
        frameData, width, height = processAfile(filename)

    frameNoDisplay = -1
    while(loop):
        if frameNoDisplay != frameNo:
            frameNoDisplay = frameNo
            offEnd, YUVPixels = getBGRFrameFromYUV420File(YUVfilename, width, height, frameNo, False)
            if width > maxDisplayWidth or height > maxDisplayHeight:
                doResize = True
            #pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
            #pixels = cv2.imread(jpegFilename )
            if showImage:
                if doResize:
                    YUVPixelsResize = cv2.resize(YUVPixels, (maxDisplayWidth, maxDisplayHeight))
                    cv2.imshow('image', YUVPixelsResize)
                else:
                    cv2.imshow('image', YUVPixels)
            if showDiff:
                if frameNo > 0:
                    offEnd2, YUVPixelsPrev = getBGRFrameFromYUV420File(YUVfilename, width, height, (frameNo-1), False)
                    diffPixels = YUVPixels - YUVPixelsPrev
                    if doResize: diffPixels = cv2.resize(diffPixels, (maxDisplayWidth, maxDisplayHeight))
                    cv2.imshow('diff', diffPixels)
            if showFrameNo:
                offEnd, frameNoPixels = getBGRFrameFromYUVOneChannelFile(FrameNofilename, width/16, height/16, frameNo, False)
                frameNoPixels = scaleToVisualise(frameNoPixels, width/16, height/16, width, height, visible=False)
                if doResize: frameNoPixels = cv2.resize(frameNoPixels, (maxDisplayWidth, maxDisplayHeight))
                cv2.imshow('FrameNo', frameNoPixels)
            if showMbNo:
                offEnd, MbNoPixels = getBGRFrameFromYUVOneChannelFile(MbNofilename, width/16, height/16, frameNo, False)
                MbNoPixels = scaleToVisualise(MbNoPixels, width/16, height/16, width, height, visible=True)
                if doResize: MbNoPixels = cv2.resize(MbNoPixels, (maxDisplayWidth, maxDisplayHeight))
                cv2.imshow('MbNo', MbNoPixels)
            if showMbMode:
                offEnd, MBModePixels = getBGRFrameFromYUVOneChannelFile(MBModefilename, width/16, height/16, frameNo, False)
                MBModePixels = scaleToVisualise(MBModePixels, width/16, height/16, width, height, visible=True, uMax=2, uMin=0)
                if doResize: MBModePixels = cv2.resize(MBModePixels, (maxDisplayWidth, maxDisplayHeight))
                cv2.imshow('MBMode', MBModePixels)
            if showQP:
                offEnd, QPPixels = getBGRFrameFromYUVOneChannelFile(QPfilename, width/16, height/16, frameNo, False)
                QPPixels = scaleToVisualise(QPPixels, width/16, height/16, width, height, visible=True)
                if doResize: QPPixels = cv2.resize(QPPixels, (maxDisplayWidth, maxDisplayHeight))
                cv2.imshow('QP', QPPixels)
            if showMV:
                #np.set_printoptions(threshold=np.nan)
                offEnd, MVPixelsX, MVPixelsY = getBGRFramesFromMVsFile(MVfilename, width/4, height/4, frameNo, False, channels=2)
                # Alternatively, if we store MVX and MVY in different files, we can simply set above channels = 1
                MVPixelsX = scaleToVisualise(MVPixelsX, width/4, height/4, width, height, visible=True)
                if doResize: MVPixelsX = cv2.resize(MVPixelsX, (maxDisplayWidth, maxDisplayHeight))
                cv2.imshow('MVX', MVPixelsX)
                MVPixelsY = scaleToVisualise(MVPixelsY, width / 4, height / 4, width, height, visible=True)
                if doResize: MVPixelsY = cv2.resize(MVPixelsY, (maxDisplayWidth, maxDisplayHeight))
                cv2.imshow('MVY', MVPixelsY)
            if showMetaData:
                metaData = turnElementsIntoFrame(frameData, frameNo, width, height, "skipped")
                if doResize: metaData = cv2.resize(metaData, (maxDisplayWidth, maxDisplayHeight))
                cv2.imshow('data', metaData)
            if showFourier:
                f = np.fft.fft2(YUVPixels)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift))

                plt.subplot(121), plt.imshow(pixels)
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(magnitude_spectrum)
                plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
                plt.show()
                loop = False

                rows, cols = height, width
                crow, ccol = rows / 2, cols / 2
                fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)

                plt.subplot(131), plt.imshow(pixels, cmap='gray')
                plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(132), plt.imshow(img_back, cmap='gray')
                plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
                plt.subplot(133), plt.imshow(img_back)
                plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

                plt.show()

        if offEnd:
            if increment < 0:
                increment = 1
            else:
                increment = -1
        if play:
            frameNo = frameNo + increment
        if frameNo < 0:
            frameNo = 0
            increment = 1
        print(frameNo)
        sys.stdout.write("\033[F")

        k = cv2.waitKey(33) & 0xFF
        if k == 27:  # Esc key to stop
            print("Quit")
            loop = False
        elif k == ord('q'):
            print("Quit")
            loop = False
        elif k == 255:  # normally -1 returned,so don't print it
            continue
        elif k == ord('p'):
            play = not play
            if play:
                print("Play")
            else:
                print("Pause")
        elif k == 3:
            frameNo = frameNo + 1
        elif k == 2:
            frameNo = frameNo - 1
            if frameNo < 0:
                frameNo = 0
        else:
            print k  # else print its value





    cv2.destroyAllWindows()

    # Create a black image
    #img = np.zeros((512, 512, 3), np.uint8)

    # Draw a diagonal blue line with thickness of 5 px
    #cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



