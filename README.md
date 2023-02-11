# Kinect Depth Motion Capture

Motion capture scripts used for *Macbeth in Space*

## Usage

1. Set up the Azure Kinect, installing the pyk4a and pygame libraries
2. Test out the camera, specifying the `MIN_DIST` and `MAX_DIST` for the captured subject alongside the `COLORMAP` of the input in `kinect.py`
3. Specify the `CAPTURE` and `FILENAME` variables in `draw.py`
4. Run `python3 draw.py`, closing the window when capturing is done
5. Convert the image folders to videos, running `ffmpeg -framerate 30 -pattern_type glob -i '<folder>/*.png' -c:v libx264 -pix_fmt yuv420p <folder>.mp4` from `images/`