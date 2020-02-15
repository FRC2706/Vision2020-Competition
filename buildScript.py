import zipfile

#create a ZipFile object
zipObj = zipfile.ZipFile('visionComp', 'w')

# Make sure to change this to the right path for your computer, or else this will not build
#path = "C:\\Vision\Vision2020-Competition" 
# Add multiple files to the zip
zipObj.write('ControlPanel.py')
zipObj.write('DistanceFunctions.py')
zipObj.write('FindBall.py')
zipObj.write('FindTarget.py')
zipObj.write('VisionConstants.py')
zipObj.write('VisionMasking.py')
zipObj.write('VisionUtilities.py')

print("I should have wrote the file")