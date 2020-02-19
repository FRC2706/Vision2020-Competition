# Vision2020-Competition
Vision repository for the competition

Contains the Merge Robotics Vision Code.

The two main files are:

MergeViewer.py: For testing the code with file images
MergeFRCPipeline.py:  Code to be run on the Raspberry Pi 4 (or 3)

To install on the Pi

Go into the FRC Pi Image, and set the Pi to 'writable'

Use the build script called 'buildScript.py' to generate the file
called visionComp.zip

>python buildScript.py

Then in the FRC 2020 Vision image, under the application tab, 
use the .zip file upload to upload the file.  Make sure the unzip option is
selected.

If an error occurs when uploading (because it won't over write automatically),
login to the pi:

>ssh pi@frcvision.local  

with password raspberry.  frcvision.local is the default name.   If the pi is on a Robot
it is probably using an IP address like 10.27.6.20 or something like that.   

the command:

>rm *.py   - This will remove all the .py images from the folder

Upload the 'visionComp.zip' file, no errors shoud occur this time

Vision code should now run on the pi!

There are network table settings under 'MergeVision' that control things like finding the Target, 
Finding the Ball, etc.  

Connect to the network table server and look for 'MergeVision' to see the settings that are available







