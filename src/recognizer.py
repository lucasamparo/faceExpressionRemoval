class Recognizer:
    '''
    Classe do Reconhecedor
    Input: Duas imagens de face
    Output: Métrica de similaridade entre elas
    Arquitetura: InceptionNet do FaceNet
    '''
    def __init__(self, inputs, reuse=None):		
        self.modelo, *_ = model(inputs, False, reuse=reuse)

    def load(self, sess):
        recog.load_model(sess, "recog")

    def computeRecog(self):
        return self.modelo