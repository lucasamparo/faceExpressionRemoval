import os, math, sys
import cv2

class Dataset:
    '''
    Dataset Class
    Input: Size of the image batch and the path to the images
    Output: Load all images and delivery the training batches
    '''
    def __init__(self, batch_size, folder='data128x128'):
        self.batch_size = batch_size
        self.include_hair = include_hair
        
        train_files = os.listdir(os.path.join(folder, 'inputs','train'))
        validation_files = os.listdir(os.path.join(folder, 'inputs','valid'))
        test_files = os.listdir(os.path.join(folder, 'inputs','test'))

        self.train_inputs, self.train_paths, self.train_targets = self.file_paths_to_images(folder, train_files)
        self.test_inputs, self.test_paths, self.test_targets = self.file_paths_to_images(folder, test_files, mode="test")
        self.valid_inputs, self.valid_paths, self.valid_targets = self.file_paths_to_images(folder, validation_files, mode="valid")

        self.pointer = 0
        self.permutation = np.random.permutation(len(self.train_inputs))
        
    def progress(self, count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    def file_paths_to_images(self, folder, files_list, mode="train"):
        inputs = []
        targets = []
        in_path = []
        test_path = []

        for count,file in enumerate(files_list):
            input_image = os.path.join(folder, 'inputs/{}'.format(mode), file)
            output_image = os.path.join(folder, 'targets/{}'.format(mode), file)
            in_path.append(os.path.join('inputs/{}'.format(mode), file))

            test_image = cv2.imread(input_image, 0)
            test_image = cv2.resize(test_image, (128, 128))
            inputs.append(test_image)

            target_image = cv2.imread(output_image, 0)
            target_image = cv2.resize(target_image, (128, 128))
            targets.append(target_image)

        return inputs, in_path, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, .15, .15)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        self.permutation = np.random.permutation(len(self.train_inputs))
        self.pointer = 0

    def next_batch(self,pretrain=False):
        inputs = []
        targets = []
        
        for i in range(self.batch_size):
            if pretrain:
                inputs.append(np.array(self.train_targets[self.permutation[self.pointer + i]]))
                targets.append(np.array(self.train_targets[self.permutation[self.pointer + i]]))
            else:
                inputs.append(np.array(self.train_inputs[self.permutation[self.pointer + i]]))
                targets.append(np.array(self.train_targets[self.permutation[self.pointer + i]]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)
