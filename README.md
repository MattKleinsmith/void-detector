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
- Label data
  - Choose a data format
    - Find and test a pipeline. Use its data format
  - Choose a bounding box hand-labeling program compatible with the chosen format
  - Convert videos to images
    - Consider minimizing the overlap of images to reduce the cost of labeling
      - Choose a good sampling rate
        - Choose fastest sampling rate, get a sense of the overlap, and choose a slower sampling rate
        - Consider using an algorithm that detects image overlap, like those used in panorama creators
  - Consider which kind of voids to label for the prototype:
    - Complete void
    - Void with product behind it
    - Void with product in front of it
- Create train, validation, and test sets
  - Consider splitting by aisle
  - Consider splitting by store
    - If so, collect data from two more stores
- Resize data for model input
  - Consider downsampling or cropping
- Train, tune HPs, test

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
