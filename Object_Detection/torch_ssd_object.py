## IMPLEMENTING A SINGLE SHOT DETECTION MODEL USING PRETRAINED SET OF AVAILABLE WEIGHTS
# Prashant Brahmbhatt (https://www.github.com/hashbanger)

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Now we define a function to do the detections
def detect(frame, net, transform):
    '''
    Inputs will be, a frame, a ssd neural network, and a transformation to be applied on the images, 
    and that will return the frame with the detector rectangle.
    '''

    height, width = frame.shape[:2] #getting the height and width of the frame
    frame_t = transform(frame)[0] # We apply the transformation to our frame
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor and reorder to GRB as the channel order
    x = Variable(x.unsqueeze(0)) #We add a fake dimension corresponding to the batch
    y = net(x) #We feed the neural network ssd with the image and get the output y
    detections = y.data # Craeting a detections tensor contained by the output y
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object or dimensions [width, height, width, height]
    for i in range(detections.size(1)): # iterating over each class
        j = 0 # we initialize the loop variable j that will correspond to the occurences of the class.
        while detections[0, i, j, 0] >= 0.6: # We take in account all the occurences of j iof the class i that have a matching score larger than 0.6
            pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the upper left corner and the lower right corner of teh image
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0),  2) # We draw a green rectangle around the detected object
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # putting the label of the class at the upper right corner
            j = j + 1 # incrementing the j to get the next occurence
    return frame # We return the frame with our detection rectangles and labels on the frame

# Creating the ssd neural network
net = build_ssd('test') # We create an object that is our neural network
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).


# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # Creating an object of the BaseTransform Class that will do the transform

#Doing the object detection on some video
reader = imageio.get_reader('funny_dog.mp4') 
fps = reader.get_meta_data()['fps'] # We get the frames per second
writer = imageio.get_writer('output.mp4', fps = fps) 
for i, frame in enumerate(reader): # Iterating on the frames of the output
    frame = detect(frame, net.eval(), transform) # we call our detect function to detect the object in the frame
    writer.append_data(frame) # We append the next frame in the output video
    print(i, "frames processed\n") # Printing the number of processed frames
writer.close()

