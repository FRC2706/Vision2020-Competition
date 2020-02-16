import zipfile

FILENAME = "visionComp.zip"

#create a ZipFile object
zipObj = zipfile.ZipFile(FILENAME, 'w')

# Add module files to the zip
zipObj.write('ControlPanel.py')
zipObj.write('DistanceFunctions.py')
zipObj.write('FindBall.py')
zipObj.write('FindTarget.py')
zipObj.write('VisionConstants.py')
zipObj.write('VisionMasking.py')
zipObj.write('VisionUtilities.py')
zipObj.write('NetworkTablePublisher.py')
zipObj.write('MergeFRCPipeline.py','uploaded.py')

print("I should have wrote the file: " + FILENAME)