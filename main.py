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

colorB = (128, 232, 70, 156, 153, 153, 30,  0,   35, 152, 180, 60,  0,  142, 70,  100, 100, 230, 32)
colorG = (64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130, 20, 0,  0,   0,   60,  80,  0,   11)
colorR = (128, 244, 70,  102, 190, 153, 250, 220, 107, 152, 70,  220, 255, 0,   0,   0,   0, 0, 119)

global isCapturing

def streamCapture(url, queueIn):
    global isCapturing
    print('Capture stream from {}'.format(url))
    video = pafy.new(url)
    stream = video.streams[2]
    cap = cv2.VideoCapture(stream.url)
    while cap.isOpened():
        ret, frame = cap.read()
        isCapturing = True

        if not ret:
            isCapturing = False
            break
        queueIn.put(frame)

    cap.release()


def display(ip, port, queueOut):
    global isCapturing
    width, height = 512, 256
    pipeline = 'appsrc ! videoconvert ! video/x-raw,format=I420,width={},height={} ! videoconvert ! jpegenc ! rtpjpegpay ! queue ! udpsink host={}  port={}'.format(width, height, ip, port)
    out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 30, (width, height), True)
    print(pipeline)

    while isCapturing:
        frame = queueOut.get()
        #print('Write frame')
        out.write(frame)

    out.release()

def label_to_pixel(img):
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    for idx, bgr in enumerate(zip(colorB, colorG, colorR)):
        result[(img == idx)] = bgr

    return result

def runSegmentation(worker, dpu, queueIn, queueout):
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
            continue
        frame = queueIn.get() 
        frame = cv2.resize(frame, (inputWidth, inputHeight), interpolation=cv2.INTER_LINEAR) 
        img = frame.astype(np.float32)
        mean = (104.0, 117.0, 123.0);
        img = img - mean

        outputData = []
        inputData  = []
        outputData.append(np.empty((batchSize, outputHeight, outputWidth, outputChannel), dtype = np.float32, order = 'C'))
        inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))

        imageRun = inputData[0]
        imageRun[0, ...] = img.reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)

        pred_label = np.argmax(outputData[0][0], axis=-1) 
        seg_img = label_to_pixel(pred_label)
        seg_img = cv2.resize(seg_img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) 
        out_img = cv2.addWeighted(frame, 0.4, seg_img, 0.6, 0)

        #print('Prediction done')
        queueout.put(out_img)


def get_subgraph(g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children if s.metadata.get_attr_str ("device") == "DPU"]
    return sub

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage {} <dpu model file>'.format(sys.argv[0]))
        exit(-1)

    model = sys.argv[1]
    g = xir.graph.Graph.deserialize(pathlib.Path(model))
    subgraphs = get_subgraph (g)
    assert len(subgraphs) == 1 # only one DPU kernel

    global isCapturing
    ip      = '192.168.3.2'
    port    = '1234'
    url     = 'https://www.youtube.com/watch?v=lkIJYc4UH60'
    threads = 2

    all_dpu_runners = [];
    for i in range(int(threads)):
        all_dpu_runners.append(runner.Runner(subgraphs[0], "run"));

    # Init synchronous queues for inter-thread communication
    queueIn = queue.Queue()
    queueOut = queue.Queue()

    # Launch threads
    threadAll = []
    sc = threading.Thread(target=streamCapture, args=(url, queueIn))
    threadAll.append(sc)

    for i in range(threads):
        tw = threading.Thread(target=runSegmentation, args=(i, all_dpu_runners[i], queueIn, queueOut))
        threadAll.append(tw)

    td = threading.Thread(target=display, args=(ip, port, queueOut))
    threadAll.append(td)

    isCapturing = True
    for x in threadAll:
        x.start()

    # Wait for all threads to stop
    for x in threadAll:
        x.join()

    # clean up resources
    for runner in all_dpu_runners:
        del runner

