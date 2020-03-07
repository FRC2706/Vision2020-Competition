from networktables import NetworkTablesInstance
from networktables import NetworkTables

def publishNumber(MergeVisionPipeLineTableName, name, value):
    #start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    # Name of network table - this is how it communicates with robot. IMPORTANT
    networkTable = NetworkTables.getTable(MergeVisionPipeLineTableName)
    networkTable.putNumber(name, value)
    #print(name+ ": " + str(value))