# DroneDetection
Assisting drone detection with human retina inspired video preprocessor.

The drone detector used is a YOLOv4 model, source code available at https://github.com/chuanenlin/drone-net. It was compiled with the AlexeyAB branch of darknet, https://github.com/AlexeyAB/darknet.

To use the model open the SplitVidsRetina notebook, put the path of the videos into the "selectedVids" array and run the notebook, the result should be two folders of images, one titled "base", one title "retina". If these folders are placed in the "data" subfolder of the drone detector, and the evaluate.py is put in the main folder of the drone detector, running evaluate.py will run the evaluation, it will print the results for each video into the console.
