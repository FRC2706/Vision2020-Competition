# Class that runs a separate thread for reading  camera server also controlling exposure.
class WebcamVideoStream:
    def __init__(self, camera, cameraServer, frameWidth, frameHeight, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream

        # Automatically sets exposure to 0 to track tape
        self.webcam = camera
        self.webcam.setExposureManual(7)
        #self.webcam.setExposureAuto()

        # Some booleans so that we don't keep setting exposure over and over to the same value
        self.autoExpose = True
        self.prevValue = True
        
        # Make a blank image to write on
        self.img = np.zeros(shape=(frameWidth, frameHeight, 3), dtype=np.uint8)
        # Gets the video
        self.stream = cameraServer.getVideo()
        (self.timestamp, self.img) = self.stream.grabFrame(self.img)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            global switch
            if self.stopped:
                return

            if switch == 1: #driver mode
                self.autoExpose = True
                ##print("Driver mode")
                if self.autoExpose != self.prevValue:
                    #self.webcam.setExposureManual(60)
                    self.webcam.setExposureManual(39)
                    self.webcam.setExposureAuto()
                    ##print("Driver mode")
                    self.prevValue = self.autoExpose
             
            elif switch == 2: #Tape Target Mode - set manual exposure to 20
                self.autoExpose = False
                if self.autoExpose != self.prevValue:
                    self.webcam.setExposureManual(7)
                    self.prevValue = self.autoExpose

            elif switch == 3: #Power Cell Mode - set exposure to 39
                self.autoExpose = False
                if self.autoExpose != self.prevValue:
                    self.webcam.setExposureManual(35)
                    self.prevValue = self.autoExpose

            # gets the image and timestamp from cameraserver
            (self.timestamp, self.img) = self.stream.grabFrame(self.img)

    def read(self):
        # return the frame most recently read
        return self.timestamp, self.img

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def getError(self):
        return self.stream.getError()

# from MergeFRCPipeline
def startCamera(config):
    #print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)
    camera.setConfigJson(json.dumps(config.config))
    return cs, camera

# class that runs separate thread for showing video,
class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """
    def __init__(self, imgWidth, imgHeight, cameraServer, frame=None):
        self.outputStream = cameraServer.putVideo("2706_out", imgWidth, imgHeight)
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.outputStream.putFrame(self.frame)

    def stop(self):
        self.stopped = True

    def notifyError(self, error):
        self.outputStream.notifyError(error)

