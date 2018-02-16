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

Next steps:
- Convert video to images.
  - Choose a good sampling rate
  - Consider minimizing the overlap of images to reduce the cost of labeling
  - Consider bounding box software options before choosing how to convert the video to images
- Label data
  - Find a bounding box labeling helper program
  - Consider which kind of voids to label for the prototype:
    - Complete void
    - Void with product behind it
    - Void with product in front of it
- Create train, validation, and test sets
  - Consider splitting by aisle
  - Consider splitting by store
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
- Output: xyz-coordinates of voids
