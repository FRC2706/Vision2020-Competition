from networktables import NetworkTablesInstance
from networktables import NetworkTables
#from networktables.util import ntproperty

def publishNumber(name, value):
    #start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    # Name of network table - this is how it communicates with robot. IMPORTANT
    networkTable = NetworkTables.getTable('MergeVision')
    networkTable.putNumber(name, value)