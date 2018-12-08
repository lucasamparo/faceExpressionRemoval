from recog_model import inception_resnet_v1 as model
import load_recog as recog

class Recognizer:
    '''
    Recognizer Class
    Input: Two face image sets
    Output: Similarity between then
    Architecture: InceptionNet from FaceNet
    '''
    def __init__(self, inputs, reuse=None):		
        self.modelo, *_ = model(inputs, False, reuse=reuse)

    def load(self, sess):
        recog.load_model(sess, "recog")

    def computeRecog(self):
        return self.modelo