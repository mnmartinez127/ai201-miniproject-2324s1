import cv2,os,random
import numpy as np
import albumentations as A
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import skimage
#TODO: Add image convolutions
#TODO: Move SMOTE and augmentation (in that order) to preprocessing step
#TODO: Replace images and filenames dictionaries with 3 lists or a pandas dataframe (not urgent)
class DataClass:
    def __init__(self, folder_name, classes=[], width=255, height=255, kernel_size=(5,5),
                 threshold_value=127, max_threshold=255,random_state=None):
        self.randomizer = np.random.default_rng(random_state)
        self.folder_name = folder_name
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.threshold_value = threshold_value
        self.max_threshold = max_threshold
        self.classes = self._get_classes(classes)
        self.images = []
        self.filenames = []
        self.labels = []

    def _get_classes(self,classes=[]):
        classlist = sorted([classname for classname in os.listdir(self.folder_name)
                            if os.path.isdir(os.path.join(self.folder_name,classname))])
        return classlist if not classes else set(classlist).intersection(set(classes))

    def read_data(self, limit=0, classes=[]):
        #Read data from an image dataset
        self.images,self.filenames,self.labels = [],[],[] #reset dataset
        self._get_classes(classes)
        ctr = 0
        #Read up to <limit> number of images from each class
        print("Reading images...")
        for classidx in range(len(self.classes)):
            classname = self.classes[classidx]
            print(f"Reading class {classname}...")
            filenames = os.listdir(os.path.join(self.folder_name,classname))
            if limit > 0: #limit the number of entries in each class
                filenames = filenames[:limit]
            for filename in filenames:
                file_path = os.path.join(self.folder_name, classname, filename)
                img = cv2.imread(file_path)
                ctr += 1
                #print(f"Reading file {ctr}: {file_path}") #Uncomment to see all files
                if img is not None:
                    self.images.append(img)
                    self.filenames.append(filename)
                    self.labels.append(classidx)
                #else: print(f"Failed to read file {ctr}: {file_path}")
        print("Reading done!")
        assert len(self.images) == len(self.filenames) == len(self.labels)
        return self.images

    def get_dataset(self,limit=0): #get the dataset as-is
        if len(self.images) == 0: #load data if not yet loaded or processed
            self.images = self.read_data(limit=limit)
        return np.array(self.images),np.array(self.labels)

    def get_filepaths(self,limit=0): #get the filepaths for the dataset
        if len(self.images) == 0: #load data if not yet loaded or processed
            self.images = self.read_data(limit=limit)
        return np.array(self.images),np.array(self.labels)

    def _k_means(self,img):
        Z = np.float32(img.reshape((-1,3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2

    def _smote_data(self,X=None,y=None,random_state=None):
        X_shape = X.shape
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],-1)#Flatten image matrix before resizing
        sm = SMOTE(random_state=random_state.integers(2**32-1))
        X,y = sm.fit_resample(X,y)
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],X_shape[1],X_shape[2],X_shape[3])#Un-flatten image matrix
        return X,y

    def _oversample_data(self,X=None,y=None,random_state=None):
        ros = RandomOverSampler(sampling_strategy="minority")
        X,y = ros.fit_resample(X,y)
        return X,y

    def _undersample_data(self,X=None,y=None,random_state=None):
        ros = RandomUnderSampler(sampling_strategy="majority")
        X,y = ros.fit_resample(X,y)
        return X,y
    
    def balance_data(self,X=None,y=None,random_state=None,apply_smote=True,apply_oversampling=False,apply_undersampling=False):
        #Use only on training set!
        if random_state is None:
            random_state = self.randomizer
        if X is None or y is None:
            X,y = self.get_dataset()
        if apply_smote:
            X,y = self._smote_data(X,y,random_state)
        if apply_oversampling:
            X,y = self._oversample_data(X,y,random_state)
        if apply_undersampling:
            X,y = self._undersample_data(X,y,random_state)
        return X,y

    def extract_features(self,image,is_gray = False):
        """ Assuming image is already preprocessed.
        """
        if not is_gray: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        # texture feature
        lbp = skimage.feature.local_binary_pattern(image, P=8, R=1)
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(257), range=(0, 256))
        lbp_hist = lbp_hist.astype('float')
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        # edge detection feature
        edges = skimage.filters.sobel(image)
        edge_area = np.sum(edges)

        # getting histogram of pixel intensities
        hist, _ = np.histogram(image, bins=256, range=(0,1))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-6)

        feature_vector = np.concatenate([lbp_hist, [edge_area], hist])

        return feature_vector


    def apply_feature_extraction(self,X,y, encode=False):
        if X is None or y is None:
            X,y = self.get_dataset()
        X = np.array([self.extract_features(image) for image in X]) #get features of each image
        if encode:
            lb = LabelBinarizer()
            y = lb.fit_transform(y)  # one-hot encode the labels
        return X, y

    def normalize_and_encode(self,X, y, encode=False):
        """ Assuming image is already preprocessed.
        """
        if X is None or y is None:
            X,y = self.get_dataset()
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1) #convert to 1D
        X = X.astype(np.float32) / 255.0  # normalize to range [0, 1]
        if encode:
            lb = LabelBinarizer()
            y = lb.fit_transform(y)  # one-hot encode the labels
        return X, y

    def encode_labels(self,y):
        if y is None:
            _,y = self.get_dataset()
        lb = LabelBinarizer()
        y = lb.fit_transform(y)  # one-hot encode the labels
        return y

    def augment_image(self,transform,image,filename,label,count=10):
        new_images,new_filenames,new_labels = [],[],[]
        for c in range(1,count+1):
            new_images.append(transform(image=image)['image'])
            new_filenames.append(f'_{c}'.join(os.path.splitext(filename)))
            new_labels.append(label)
        return new_images,new_filenames,new_labels

    def augment_images(self,count=10):
        self.get_dataset() #load data if not yet loaded
        #Pipeline used in augmentations
        transform = A.Compose([
        #flip and randomly scale and rotate the image
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.625, scale_limit=0.1, rotate_limit=60, interpolation = 3, border_mode = 3, p=0.8),
        ])
        ctr = 0
        print("Augmenting images...")
        new_images,new_filenames,new_labels = [],[],[]
        for idx in range(len(self.images)):
            new_images_i,new_filenames_i,new_labels_i = self.augment_image(transform,self.images[idx],self.filenames[idx],self.labels[idx],count)
            new_images.extend(new_images_i)
            new_filenames.extend(new_filenames_i)
            new_labels.extend(new_labels_i)
            ctr += 1
            #print(f'Augmented file {ctr}: {self.filenames[idx]}')
            #add new images after processing all images in a class
            self.images.extend(new_images)
            self.filenames.extend(new_filenames)
            self.labels.extend(new_labels)
        print("Data augmentation complete!")

    def process_images(self, resize=True, convert_to_grayscale=False,
                        apply_filter=False, segment_images=False, k_means = False,
                          display_images=False, limit=0):
        print("Processing images...")
        self.get_dataset(limit) #load data if not yet loaded
        ctr = 0
        processed_images = []
        assert len(self.images) == len(self.filenames) == len(self.labels)
        for idx in range(len(self.images)):
            #process image and generate segmentation masks separately
            processed_img = np.copy(self.images[idx])
            if resize:
                processed_img = cv2.resize(processed_img, (self.width, self.height),interpolation=cv2.INTER_CUBIC)
            k_img = self._k_means(processed_img)
            gray_img = cv2.cvtColor(k_img, cv2.COLOR_BGR2GRAY)
            blur_img = cv2.GaussianBlur(gray_img, self.kernel_size, 0)
            _, segmented_mask = cv2.threshold(blur_img,self.threshold_value,self.max_threshold,cv2.THRESH_BINARY)
            segmented_img = cv2.bitwise_and(processed_img,processed_img,mask=cv2.bitwise_not(segmented_mask))
            if k_means:
                processed_img = self._k_means(processed_img)
            if convert_to_grayscale:
                processed_img= cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            if apply_filter:
                processed_img=cv2.GaussianBlur(processed_img, self.kernel_size, 0)
            if segment_images:
                processed_img = cv2.bitwise_and(processed_img,processed_img,mask=cv2.bitwise_not(segmented_mask))
            if display_images:
                cv2.imshow(f"Original Image", self.images[idx])
                cv2.imshow(f"Gray Image", gray_img)
                cv2.imshow(f"Blurred Image", blur_img)
                cv2.imshow(f"Segmentation Mask", segmented_mask)
                cv2.imshow(f"Segmented Image", segmented_img)
                cv2.imshow(f"Processed Image", processed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            processed_images.append(processed_img)
            ctr += 1
            print(f'Processed file {self.classes[self.labels[idx]]}-{ctr}: {self.filenames[idx]}')
        assert len(self.images) == len(processed_images)
        self.images = processed_images
        print("Image processing complete!")

    def show_images(self, start, stop=None):
        self.get_dataset()
        assert len(self.images) == len(self.filenames) == len(self.labels)
        if not stop: start,stop = 0,start
        for classidx in self.classes:
            print(f"Showing images of class {classidx}")
            imageidxs = sorted([idx for idx in range(len(self.labels)) if self.labels[idx] == classidx])[start:stop]
            for i,idx in enumerate(imageidxs):
                cv2.imshow(f"Image {self.classes[classidx]}-{i+1}: {self.filenames[idx]}", self.images[idx])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def write_data(self, out_path='datasets/processed'):
        self.get_dataset()
        assert len(self.images) == len(self.filenames) == len(self.labels)
        print(f'Creating folder for processed dataset...')
        os.makedirs(out_path,exist_ok=True)
        ctr = 0
        for classidx,classname in enumerate(self.classes):
            dir_path = os.path.join(out_path, classname)
            os.makedirs(dir_path,exist_ok=True)
            imageidxs = sorted([idx for idx in range(len(self.labels)) if self.labels[idx] == classidx])
            for idx in imageidxs:
                file_path = os.path.join(dir_path, self.filenames[idx])
                ctr += 1
                print(f'Writing file {ctr}: {self.filenames[idx]}->{file_path}')
                cv2.imwrite(file_path, self.images[idx])

    def write_split_data(self,out_path='datasets/split',test_size=None,train_size=None,random_state = None,shuffle = True,stratify=None):
        self.get_dataset()
        assert len(self.images) == len(self.filenames) == len(self.labels)
        #Splits the image paths to save training and testing sets in separate folders
        train_idx,test_idx, = train_test_split(range(len(self.images)),test_size=0.7,train_size=0.3,random_state=random_state,shuffle=shuffle,stratify=stratify)

        print(f'Creating folder for split dataset...')
        os.makedirs(out_path,exist_ok=True)
        ctr = 0
        for classidx,classname in enumerate(self.classes):
            dir_train_path = os.path.join(out_path, "train", classname)
            dir_test_path = os.path.join(out_path, "test", classname)
            dir_other_path = os.path.join(out_path, "other", classname)
            os.makedirs(dir_train_path,exist_ok=True)
            os.makedirs(dir_test_path,exist_ok=True)
            os.makedirs(dir_other_path,exist_ok=True)
            imageidxs = sorted([idx for idx in range(len(self.labels)) if self.labels[idx] == classidx])
            for idx in imageidxs:
                if idx in train_idx:
                    file_path = os.path.join(dir_train_path, self.filenames[idx])
                elif idx in test_idx:
                    file_path = os.path.join(dir_test_path, self.filenames[idx])
                else:
                    file_path = os.path.join(dir_other_path, self.filenames[idx])
                ctr += 1
                print(f'Writing file {ctr}: {self.filenames[idx]}->{file_path}')
                cv2.imwrite(file_path, self.images[idx])

if __name__ == "__main__":
 
    random_state = 13535
    train_aug_count,test_aug_count = 5,5
    limit=5
    raw_path = 'datasets/Coconut Tree Disease Dataset/'
    processed_path = 'datasets/processed/processed/'
    split_path = 'datasets/split/'
    processed_train_path = 'datasets/processed/train/'
    processed_test_path = 'datasets/processed/test/'
    #Read and process the dataset

    data = DataClass(folder_name=raw_path,random_state=random_state)
    data.read_data(limit=limit)
    data.process_images(resize=True)
    data.write_data(processed_path)
    data.write_split_data(split_path,test_size=0.3,train_size=0.7,random_state=random_state,shuffle=True if random_state is not None else False,stratify=data.labels)
    print("Split complete!")
    train_data = DataClass(folder_name=os.path.join(split_path,"train/"))
    test_data = DataClass(folder_name=os.path.join(split_path,"test/"))
    train_data.read_data()
    test_data.read_data()
    train_data.augment_images(train_aug_count)
    test_data.augment_images(test_aug_count)
    train_data.write_data(processed_train_path)
    train_data.write_data(processed_test_path)
    train_data = DataClass(folder_name=processed_train_path)
    test_data = DataClass(folder_name=processed_test_path)
    train_data.read_data()
    test_data.read_data()

    X_train,y_train = train_data.get_dataset()
    print(X_train.shape)
    print(y_train.shape)
    X_train1,y_train1 = train_data.apply_feature_extraction(X_train,y_train) #contains features
    X_train2,y_train2 = train_data.normalize_and_encode(X_train,y_train) #contains normalized images
    X_train1,y_train1 = train_data.balance_data(X_train1,y_train1,apply_smote=True) #balance dataset
    X_train2,y_train2 = train_data.balance_data(X_train2,y_train2,apply_smote=True) #balance dataset

    X_test,y_test = test_data.get_dataset()
    z_test = np.array(test_data.filenames)
    print(X_test.shape)
    print(y_test.shape)
    print(z_test.shape)
    X_test1,y_test1 = test_data.apply_feature_extraction(X_test,y_test)
    X_test2,y_test2 = test_data.normalize_and_encode(X_test,y_test)
