import numpy as np
import cv2
import depthai as dai
import blobconverter as bcr


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def main():
    stream = dai.Pipeline()

    cam_rgb = stream.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)

    detection_nn = stream.create(dai.node.MobileNetDetectionNetwork)
    
    detection_nn.setBlobPath(bcr.from_zoo(name='mobilenet-ssd', shaves=6))
    detection_nn.setConfidenceThreshold(0.5)

    cam_rgb.preview.link(detection_nn.input)

    xout_rgb = stream.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_nn = stream.create(dai.node.XLinkOut)
    
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)

    with dai.Device(stream,) as oakd:
        q_rgb = oakd.getOutputQueue("rgb")
        q_nn = oakd.getOutputQueue("nn")
        frame = None
        detections = []

        while 1:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()
            if in_rgb is not None: frame = in_rgb.getCvFrame()
            if in_nn is not None: detections = in_nn.detections
            if frame is not None:
                for detection in detections:
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)  
                cv2.imshow("preview", frame)
            if cv2.waitKey(1) == ord('q'): break


def print_hi(name):
    print(f'Hi {name}!') 


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Kerry')
    main()


