import subprocess
import os
import signal

"""
this file runs the object detector on two folder 
(one for the base unprocessed images and one for the processed images) 
it outputs the average precision (AP) for each video in each folder
"""


def runDarknet(path: str):
    command = [
        "./darknet",
        "detector",
        "test",
        "cfg/drone.data",
        "cfg/yolo-drone.cfg",
        "weights/yolo-drone.weights",
        "-thresh",
        "0.25",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print("running darknet on all images in", path)
    image_files = [  # List all image files in the folder
        f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    input_data = (
        "\n".join(os.path.join(path, file_name) for file_name in image_files) + "\n"
    )

    process.stdin.write(input_data)
    process.stdin.flush()
    print("waiting for darknet")
    output, error = process.communicate()
    process.send_signal(signal.SIGINT)  # Send Ctrl+C signal to interrupt the C program

    if error:
        print("darknet error messages:")
        print(error)
    return output


def convertToLib(consoleLog):
    print("parsing darknet output")
    lines = consoleLog.split("\n")
    name = ""
    confidence = 0
    bounding_boxes = {}
    for line in lines:
        if line[0:5] == "data/":
            # Console out looks like
            # Enter Image Path: *NAME*: Predicted in 0.522983 seconds.
            name = line.split(" ")[0][:-1]
            bounding_boxes[name] = []
            print(name)
        if line[0:5] == "Drone":
            # Console out looks like
            # Drone: XX%
            confidence = float(line[7:9]) / 100
        if line[0:5] == "Bound":
            # Console out looks like
            # Bounding Box: Left=*LEFT*, Top=*TOP*, Right=*RIGHT*, Bottom=*BOTTOM*
            words = line.split(" ")
            boundingBox = [
                0,
                int(words[2][5:-1]),
                int(words[3][4:-1]),
                int(words[4][6:-1]),
                int(words[5][7:]),
                confidence,
            ]
            bounding_boxes[name] += boundingBox
    return bounding_boxes


def compareBB(bbs1, bbs2):
    detections1 = len(bbs1)
    detections2 = len(bbs2)
    if detections1 > 0 and detections2 > 0:
        return [1, 0, 0, 0]
    if detections1 > 0:
        return [0, 1, 0, 0]
    if detections2 > 0:
        return [0, 0, 1, 0]
    return [0, 0, 0, 1]


def addData(y, x):
    y[0] += x[0]
    y[1] += x[1]
    y[2] += x[2]
    y[3] += x[3]
    pass


def compare(lib1, lib2, path1, path2):
    y = [0, 0, 0, 0]
    for nm, bbs in lib1.items():
        print(nm)
        s = nm.split("/")
        if len(s) == 4:
            img = nm.split("/")[3]
            x = compareBB(lib1[path1 + "/" + img], lib2[path2 + "/" + img])
            addData(y, x)
    return y


"""
I need to load the real labels and compare the bounding boxes with them!
"""


def IoU(bb1, bb2):
    # using bb notation as x,y,w,h
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    if not (x1 + w1 > x2 and x2 + w2 > x1 and y1 + h1 > y2 and y2 + h2 > y1):
        return 0.0  # do not intersect
    left: float = max(x1, x2)
    right: float = min(x1 + w1, x2 + w2)
    width: float = right - left
    top: float = max(y1, y2)
    bottom: float = min(y1 + h1, y2 + h2)
    height: float = bottom - top
    intersection: float = width * height
    union: float = w1 * h1 + w2 * h2 - intersection
    return intersection / union


class ConfusionMatrix:
    def __init__(self):
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0
        pass

    def precision(self):
        if self.TP + self.FP == 0:
            return 0
        return self.TP / (self.TP + self.FP)

    def recall(self):
        if self.TP + self.FN == 0:
            return 0
        return self.TP / (self.TP + self.FN)

    def addToConfusionMatrix(self, predBB, realBB, IOUTHRESHOLD):
        if predBB is not None and realBB is None:  # FP
            self.FP += 1
        elif realBB is not None and predBB is None:  # FN
            self.FN += 1
        elif realBB is None and predBB is None:  # TN
            self.TN += 1
        elif IoU(predBB, realBB) > IOUTHRESHOLD:  # TP
            self.TP += 1
        else:  # FP
            self.FP += 1
        pass


def PrecisionRecall(mat):
    return [mat.precision(), mat.recall()]


def PRCurve(predBBsList, realBBsList, thresholds):
    PR = []
    for threshold in thresholds:
        mat = ConfusionMatrix()
        for j in range(len(predBBsList)):
            mat.addToConfusionMatrix(predBBsList[j], realBBsList[j], threshold)
        PR.append([mat.precision(), mat.recall()])
    return PR


def AP(predBBsList, realBBsList):
    prList = PRCurve(predBBsList, realBBsList, [0.5, 0.6, 0.7, 0.8, 0.9])
    ap = 0
    last_recall = 0
    prList.sort(key=lambda x: x[1])
    for precision, recall in prList:
        ap += precision * (recall - last_recall)
        last_recall = recall
    return ap


def AUC_PR(predBBsList, realBBsList):
    k = 100
    prList = PRCurve(predBBsList, realBBsList, [i / k for i in range(k + 1)])
    auc = 0
    last_recall = 0
    prList.sort(key=lambda x: x[1])
    for precision, recall in prList:
        auc += precision * (recall - last_recall)
        last_recall = recall
    return auc


def getBBsPerVid(lib):
    print("extracting bounding boxxes")
    PredBBs = {}  # ordered list of BBs
    RealBBs = {}  # ordered list of BBs
    for nm, bbs in lib.items():
        s = nm.split("/")
        if len(s) > 1:
            # path looks like data/SOMETHING/FILE_XXXX.jpg where XXXX is the frame number with leading zeros
            imageFile = nm.split("/")[2]  # Extract File_XXXX.jpg
            frameNumber = int(imageFile[-8:-4])  # Extract XXXX
            fileName = imageFile[:-9]  # Extract File

            with open(os.path.join("data", "annotations", fileName + ".txt"), "r") as f:
                annotations = f.read().split("\n")
            if len(annotations) <= frameNumber:
                continue
            anno = annotations[frameNumber].split(" ")
            if len(anno) < 3:
                realBB = None
            else:
                realBB = [int(anno[2]), int(anno[3]), int(anno[4]), int(anno[5])]
            if len(bbs) == 0:
                predBB = None
            else:
                # adjustments may need to be made to this BB
                predBB = [bbs[1], bbs[2], bbs[3] - bbs[1], bbs[4] - bbs[2]]
            if fileName not in PredBBs:
                PredBBs[fileName] = [predBB]
                RealBBs[fileName] = [realBB]
            else:
                PredBBs[fileName] += [predBB]
                RealBBs[fileName] += [realBB]
    return [PredBBs, RealBBs]


# framenum num_objs_in_frame obj1_x_left obj1_y_top obj1_w obj1_h obj1_class ...
def getBBs(lib):
    print("extracting bounding boxxes")
    PredBBs = []  # ordered list of BBs
    RealBBs = []  # ordered list of BBs
    for nm, bbs in lib.items():
        s = nm.split("/")
        if len(s) > 1:
            # path looks like data/SOMETHING/FILE_XXXX.jpg where XXXX is the frame number with leading zeros
            imageFile = nm.split("/")[2]  # Extract File_XXXX.jpg
            frameNumber = int(imageFile[-8:-4])  # Extract XXXX
            fileName = imageFile[:-9]  # Extract File
            with open(os.path.join("data", "annotations", fileName + ".txt"), "r") as f:
                annotations = f.read().split("\n")
            if len(annotations) <= frameNumber:
                continue
            anno = annotations[frameNumber].split(" ")
            if len(anno) < 3:
                realBB = None
            else:
                realBB = [int(anno[2]), int(anno[3]), int(anno[4]), int(anno[5])]
            if len(bbs) == 0:
                predBB = None
            else:
                # adjustments may need to be made to this BB
                predBB = [bbs[1], bbs[2], bbs[3] - bbs[1], bbs[4] - bbs[2]]
            PredBBs += [predBB]
            RealBBs += [realBB]
    return [PredBBs, RealBBs]


def getMetrics(selection):
    path = os.path.join("data", selection)
    consoleLog = runDarknet(path)
    lib = convertToLib(consoleLog)
    predicted, real = getBBs(lib)
    return AP(predicted, real), AUC_PR(predicted, real)


def printBBsPerVid(selection):
    path = os.path.join("data", selection)
    consoleLog = runDarknet(path)
    lib = convertToLib(consoleLog)
    predicted, real = getBBsPerVid(lib)
    print(predicted, real)
    pass


def getMetricsPerVidFromLib(lib):
    predicted, real = getBBsPerVid(lib)
    mAPs = {}
    AUCs = {}
    for file, _ in predicted.items():
        mAPs[file] = AP(predicted[file], real[file])
        AUCs[file] = AUC_PR(predicted[file], real[file])
    return mAPs, AUCs


def getLib(selection):
    return convertToLib(runDarknet(os.path.join("data", selection)))


def getMetricsPerVid(selection):
    return getMetricsPerVidFromLib(getLib(selection))

# get metrics and prints them out
if __name__ == "__main__":
    _, base_metrics = getMetricsPerVid("base")
    _, retina_metrics = getMetricsPerVid("retina")
    print("base")
    print(f"VIDEO\t|\tAP")
    for file, metric in base_metrics.items():
        print(f"{file}\t|\t{round(metric,4)}")
    print("retina")
    print("VIDEO\t|\tAP")
    for file, metric in retina_metrics.items():
        print(f"{file}\t|\t{round(metric,4)}")
        
# old version which prints out metrics for base and retina but doesn't divide it by video
if not __name__ == "__main__":
    mAP_base, AUR_PR_base = getMetrics("base")
    mAP_retina, AUR_PR_retina = getMetrics("retina")
    print(
        f"""
          TYPE|\tBASE\t|RETINA
          mAP |\t{round(mAP_base,4)}\t|{round(mAP_retina,4)}
          AUC |\t{round(AUR_PR_base,4)}\t|{round(AUR_PR_retina,4)}
          """
    )
