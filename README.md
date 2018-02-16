# void-detector

Detect voids in grocery store shelves.

Work in progress.

# Goal

- Input: Image of shelf
- Output: Bounding boxes around voids

# Log

2018-02-15: Collect data.
- I walked down the aisles of my local grocery store while recording the shelves.
- I used a grocery cart to stable my smartphone's camera and to bring the height of the camera close to the height of Focal Systems' camera.
- I used the right lane of each aisle and recorded the shelves of the opposite lane.
- I walked through each aisle twice to record each side of each aisle.
- The video had dimensions 640 x 480 at 30 fps.
- It took 14 minutes to scan the store.

Plan:
- [ ] Label data
  - [x] Choose a data format
    - [x] Find and test a pipeline. Use its data format
      - Pipeline chosen: [torchcv](https://github.com/kuangliu/torchcv/)
      - Data format: lines in train.txt: name.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
  - [x] Choose a bounding box hand-labeling program compatible with the chosen format
    - Labeler chosen: [YOLO v2 Bounding Box Tool](https://github.com/Cartucho/yolo-boundingbox-labeler-GUI)
      - <img src="labeler.png" width="40%">
    - [My fork](https://github.com/MattKleinsmith/yolo-boundingbox-labeler-GUI/tree/patch-2)
        - [x] Change output to match the torchcv format
        - [x] Make bounding box colors consistent between labeling sessions
        - [x] Make box form upon mouse-up, instead of requiring two clicks
        - [x] Add filename to help screen to make debugging easier
        - [x] Add the option to display the images in order
        - [ ] Start session at first unlabeled image
        - [ ] Allow the user to start with the bottom right corner
        - [ ] Allow the user to adjust the line width
  - [x] Convert videos to images
    - [x] Consider minimizing the overlap of images to reduce the cost of labeling
      - [x] Choose a good sampling rate
        - [x] Choose fastest sampling rate, get a sense of the overlap, and choose a slower sampling rate
        - [x] Consider using an algorithm that detects image overlap, like those used in panorama creators
          - Unneeded. Manual inspection worked.
        - Sampling rate chosen: 1 fps.
        - I preserved the frame IDs with respect to 30 fps to ease the use of object detection later.
  - [x] Consider which kind of voids to label for the prototype:
    - [Yes] Complete void
    - [Yes] Void with product behind it
    - [Not yet] Void with product in front of it
- [ ] Create train, validation, and test sets
  - Consider splitting by aisle
  - Consider splitting by store
    - If so, collect data from two more stores
- [ ] Resize data for model input
  - Consider downsampling or cropping
- [ ] Train, tune HPs, test

# Ambitions

Real-time processing:
- Input: Stream of images
- Output: Stream of bounding boxes

Void categorization:
- Input: Image of shelf
- Output: Product IDs of voids

Void localization:
- Input: Image of shelf
- Output: xyz-coordinates of voids, with respect to a 3D store map

Efficient hand-labeling:
- Label a void in one frame, then use an object tracker to label the void for the rest of the frames. This would multiply the number of labels by about 30, assuming a 30 FPS camera and an on-screen-time of one second.
