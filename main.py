from ctypes import *
import pathlib
import cv2
import pafy
import numpy as np
import runner
import xir.graph
import xir.subgraph
import os
import sys
import threading
import queue
import time

colorB = (128, 232, 70, 156, 153, 153, 30,  0,   35, 152, 180, 60,  0,  142, 70,  100, 100, 230, 32)
colorG = (64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130, 20, 0,  0,   0,   60,  80,  0,   11)
colorR = (128, 244, 70,  102, 190, 153, 250, 220, 107, 152, 70,  220, 255, 0,   0,   0,   0, 0, 119)

global isCapturing

def streamCapture(stream, queueIn):
    global isCapturing
    print('Capture stream from {}'.format(url))
    cap = cv2.VideoCapture(stream.url)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        isCapturing = True

        if not ret:
            isCapturing = False
            break
        queueIn.put((frame_id, frame))
        frame_id = frame_id + 1

    cap.release()


def outputStream(ip, port, queueOut):
    global isCapturing
    width, height = 512, 256
    pipeline = 'appsrc ! videoconvert ! video/x-raw,format=I420,width={},height={} ! videoconvert ! jpegenc ! rtpjpegpay ! queue ! udpsink host={}  port={}'.format(width, height, ip, port)
    out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 30, (width, height), True)
    print(pipeline)
    while isCapturing:
        frame_id, (img_ori, pred_labels) = queueOut.get()
        seg_mask = label_to_pixel(pred_labels)
        seg_mask = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST) 
        # overlay original image with segmentation mask
        img_out = cv2.addWeighted(img_ori, 0.4, seg_mask, 0.6, 0)

        prev = frame_id 
        out.write(img_out)

    out.release()

def label_to_pixel(img):
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    for idx, bgr in enumerate(zip(colorB, colorG, colorR)):
        result[(img == idx)] = bgr

    return result

def normalize(img):
    mean = (104.0, 117.0, 123.0);
    img  = img.astype(np.float32)
    img  = img - mean
    return img

def runSegmentation(worker, dpu, queueIn, queueOut):
    global isCapturing
    print('Worker: {}'.format(worker))
    inputTensors  = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    inputHeight   = inputTensors[0].dims[1]
    inputWidth    = inputTensors[0].dims[2]
    outputHeight  = outputTensors[0].dims[1]
    outputWidth   = outputTensors[0].dims[2]
    outputChannel = outputTensors[0].dims[3]
    batchSize     = inputTensors[0].dims[0]
    outputSize    = outputHeight * outputWidth * outputChannel
    shapeIn       = (batchSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])

    #print(outputWidth, outputHeight)
    while isCapturing:
        if queueIn.empty():
            time.sleep(0.2)
            continue
        frame_id, img_ori = queueIn.get() 
        #img_ori = cv2.resize(img_ori, (inputWidth, inputHeight), interpolation=cv2.INTER_LINEAR) 
        img_ori = img_ori[60:60+inputHeight, 0:inputWidth]
        img     = normalize(img_ori)

        outputData = []
        inputData  = []
        outputData.append(np.empty((batchSize, outputHeight, outputWidth, outputChannel), dtype = np.float32, order = 'C'))
        inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))

        imageRun = inputData[0]
        imageRun[0, ...] = img.reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)

        pred_labels = np.argmax(outputData[0][0], axis=-1) 
        queueOut.put((frame_id, (img_ori, pred_labels)))
        #print('Prediction done')


def get_subgraph(g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children if s.metadata.get_attr_str ("device") == "DPU"]
    return sub

if __name__ == "__main__":
    # Change variables below for your setup
    threads = 2
    ip      = '192.168.3.2'
    port    = '1234'
    url     = 'https://www.youtube.com/watch?v=lkIJYc4UH60'
    model   = 'model/fpn.elf' 

    g = xir.graph.Graph.deserialize(pathlib.Path(model))
    subgraphs = get_subgraph (g)
    assert len(subgraphs) == 1 

    dpu_runners = [];
    for i in range(int(threads)):
        dpu_runners.append(runner.Runner(subgraphs[0], "run"));

    # Init synchronous queues for inter-thread communication
    queueIn  = queue.Queue()
    queueOut = queue.PriorityQueue()

    # Launch threads
    threadAll = []
    video = pafy.new(url)
    stream = video.streams[2]
    taskCapture = threading.Thread(target=streamCapture, args=(stream, queueIn))
    threadAll.append(taskCapture)

    for i in range(threads):
        taskPrediction = threading.Thread(target=runSegmentation, args=(i, dpu_runners[i], queueIn, queueOut))
        threadAll.append(taskPrediction)

    taskDisplay = threading.Thread(target=outputStream, args=(ip, port, queueOut))
    threadAll.append(taskDisplay)

    global isCapturing
    isCapturing = True

    for t in threadAll:
        t.start()

    # Wait for all threads to stop
    for t in threadAll:
        t.join()

    # clean up resources
    for runner in dpu_runners:
        del runner

