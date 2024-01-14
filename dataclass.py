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
    def __init__(self, folder_name, classes=[], width=256, height=256, kernel_size=(5,5),
                 threshold_value=127, max_threshold=255):
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

    def read_data(self, limit=0, classes=[],force_reload=False):
        if force_reload or len(self.images) == 0 or not(len(self.images)==len(self.labels)==len(self.filenames)):
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
                    #print(f"Reading file {ctr}: {file_path}\r",end='') #Uncomment to see all files
                    if img is not None:
                        self.images.append(img)
                        self.filenames.append(filename)
                        self.labels.append(classidx)
                    #else: print(f"Failed to read file {ctr}: {file_path}")
            print("\nReading done!")
            assert len(self.images) == len(self.filenames) == len(self.labels)
        return self.images

    def get_dataset(self,limit=0,force_reload=False): #get the dataset as-is
        self.read_data(limit=limit,force_reload=force_reload) #reload if empty
        return np.array(self.images),np.array(self.labels)

    def get_filepaths(self,limit=0,force_reload=False): #get the filepaths for the dataset
        self.images = self.read_data(limit=limit,force_reload=force_reload) #reload if empty
        return np.array(self.filenames)

    def _k_means(self,img,K=3):
        Z = np.float32(img.reshape((-1,3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2

    def _smote_data(self,X=None,y=None,random_state=None):
        random_state = np.random.default_rng(random_state)
        X_shape = X.shape
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],-1)#Flatten image matrix before resizing
        sm = SMOTE(random_state=random_state.integers(2**32-1))
        X,y = sm.fit_resample(X,y)
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],X_shape[1],X_shape[2],X_shape[3])#Un-flatten image matrix
        return X,y

    def _oversample_data(self,X=None,y=None,random_state=None):
        ros = RandomOverSampler(sampling_strategy="auto",random_state=random_state)
        X_shape = X.shape
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],-1)#Flatten image matrix before resizing
        X,y = ros.fit_resample(X,y)
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],X_shape[1],X_shape[2],X_shape[3])#Un-flatten image matrix
        return X,y

    def _undersample_data(self,X=None,y=None,random_state=None):
        ros = RandomUnderSampler(sampling_strategy="auto",random_state=random_state)
        X_shape = X.shape
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],-1)#Flatten image matrix before resizing
        X,y = ros.fit_resample(X,y)
        if len(X_shape) > 2:
            X = X.reshape(X.shape[0],X_shape[1],X_shape[2],X_shape[3])#Un-flatten image matrix
        return X,y
    
    def balance_data(self,X=None,y=None,random_state=None,apply_smote=False,apply_oversampling=False,apply_undersampling=False):
        #Use only on training set!
        random_state = random_state if random_state is not None else np.random.default_rng().integers(0,2**32-1)
        print(f"Random state used: {random_state}")
        if X is None or y is None:
            X,y = self.get_dataset()
        if apply_smote:
            X,y = self._smote_data(X,y,random_state)
        if apply_oversampling:
            X,y = self._oversample_data(X,y,random_state)
        if apply_undersampling:
            X,y = self._undersample_data(X,y,random_state)
        return X,y

    def extract_features(self,image,bins=64,colorspace="hsv",use_lbp=True):
        #assume BGR image is passed
        # getting color histogram of pixel intensities
        match colorspace:
            case "hsv":
                color_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                color_range = ((0,179),(0,255),(0,255))
            case "rgb":
                color_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                color_range = ((0,255),(0,255),(0,255))
            case "cielab":
                color_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
                color_range = ((0,255),(0,255),(0,255))
            case "gray":
                color_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                color_range = ((0,255),)
            case _: #pass default BGR image. Equivalent to RGB.
                color_image = image
                color_range = ((0,255),(0,255),(0,255))
        
        image_planes = [color_image] if len(color_image.shape) < 2 else cv2.split(color_image)
        #get histograms for each channel
        hists = []
        for i in range(len(image_planes)):
            hist_i,bin_edges = np.histogram(image_planes[i].flatten(),bins=bins,range=color_range[i])
            hist_i = hist_i.astype(np.float32)
            if hist_i.sum() != 0: hist_i /= hist_i.sum() #normalize histogram
            hists.append(hist_i)
        if use_lbp:
            gray_image = image if (len(image.shape) <= 2 or image.shape[2] == 1) else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale if not already
            # texture feature
            lbp = skimage.feature.local_binary_pattern(gray_image, P=8, R=1)
            lbp_bins = int(lbp.max()+1)
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=lbp_bins, range=(0, lbp_bins))
            lbp_hist = lbp_hist.astype(np.float32)
            if lbp_hist.sum() != 0: lbp_hist /= lbp_hist.sum() #normalize histogram
            hists.append(lbp_hist)
        feature_vector = np.concatenate(hists)
        if feature_vector.sum() != 0: feature_vector /= feature_vector.sum() #normalize feature vector
        return feature_vector


    def apply_feature_extraction(self,X,y,bins=64,colorspace="hsv",use_lbp=True, encode=False):
        if X is None or y is None:
            X,y = self.get_dataset()
        features = []
        for i,image in enumerate(X):
            features.append(self.extract_features(image,bins=bins,colorspace=colorspace,use_lbp=use_lbp)) #get features of each image
            print(f'Extracted features {i+1}/{len(X)}:\r',end='')
        if encode:
            lb = LabelBinarizer()
            y = lb.fit_transform(y)  # one-hot encode the labels
        print("\nFeature extraction complete!")
        return np.array(features), y

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
        self.read_data() #load data if not yet loaded
        if count <= 0:
            return #do not augment
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
            #print(f'Augmented file {ctr}: {self.filenames[idx]}\r',end='')
            #add new images after processing all images in a class
            self.images.extend(new_images)
            self.filenames.extend(new_filenames)
            self.labels.extend(new_labels)
        print("\nData augmentation complete!")

    def process_images(self, resize=True, convert_to_grayscale=False,
                        apply_filter=False, segment_images=False,
                          display_images=False):
        print("Processing images...")
        self.read_data() #load data if not yet loaded
        ctr = 0
        processed_images = []
        assert len(self.images) == len(self.filenames) == len(self.labels)
        for idx in range(len(self.images)):
            #process image and generate segmentation masks separately
            processed_img = np.copy(self.images[idx])
            if resize:
                processed_img = cv2.resize(processed_img, (self.width, self.height),interpolation=cv2.INTER_CUBIC)
            k_img = self._k_means(processed_img,K=8)
            gray_img = cv2.cvtColor(k_img, cv2.COLOR_BGR2GRAY)
            blur_img = cv2.GaussianBlur(gray_img, self.kernel_size, 0)
            _, segmented_mask = cv2.threshold(blur_img,self.threshold_value,self.max_threshold,cv2.THRESH_BINARY)
            segmented_img = cv2.bitwise_and(processed_img,processed_img,mask=cv2.bitwise_not(segmented_mask))
            #if k_means:
            #    processed_img = self._k_means(processed_img,k_means)
            if convert_to_grayscale:
                processed_img= cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            #if apply_filter:
            #    processed_img=cv2.GaussianBlur(processed_img, self.kernel_size, 0)
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
            print(f'Processed file {self.classes[self.labels[idx]]}-{ctr}: {self.filenames[idx]}\r',end='')
        assert len(self.images) == len(processed_images)
        self.images = processed_images
        print("\nImage processing complete!")

    def show_images(self, start, stop=None):
        self.read_data()
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
        self.read_data()
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
                #print(f'Writing file {ctr}: {self.filenames[idx]}->{file_path}\r',end='')
                cv2.imwrite(file_path, self.images[idx])
        print("\nWriting data complete!")

    def write_split_data(self,out_path='datasets/split',test_size=None,train_size=None,random_state = None,shuffle = True,stratify=None):
        self.read_data()
        assert len(self.images) == len(self.filenames) == len(self.labels)
        #Splits the image paths to save training and testing sets in separate folders
        train_idx,test_idx, = train_test_split(range(len(self.images)),test_size=test_size,train_size=train_size,random_state=random_state,shuffle=True if random_state is not None else shuffle,stratify=stratify)

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
                #print(f'Writing file {ctr}: {self.filenames[idx]}->{file_path}\r',end='')
                cv2.imwrite(file_path, self.images[idx])
        print("\nWriting split data complete!")

if __name__ == "__main__":
    train_aug_count,test_aug_count = 1,1
    limit=0 #set to 0 for the actual evaluation
    data_path = "datasets/"
    results_path = "results/"
    raw_path =  os.path.join(data_path,"Coconut Tree Disease Dataset/")
    split_path = os.path.join(data_path,"split/")
    processed_path = os.path.join(data_path,"processed/processed/")
    processed_train_path = os.path.join(data_path,"processed/processed/train/")
    processed_test_path = os.path.join(data_path,"processed/processed/test/")
    os.makedirs(raw_path,exist_ok=True)
    os.makedirs(results_path,exist_ok=True)
    #Read and process the dataset
    #random_state = np.random.randint((2**31)-1)
    random_state = 317401096 
    print(f"Random state used: {random_state}")

    if not os.path.exists(processed_path):
        data = DataClass(folder_name=raw_path,random_state=random_state)
        data.read_data(limit=limit)
        data.process_images(resize=True,segment_images=False)
        data.write_data(processed_path)
        data.write_split_data(split_path,test_size=0.3,train_size=0.7,random_state=random_state,shuffle=True if random_state is not None else False,stratify=data.labels)
        print("Split complete!")
    #data = DataClass(folder_name=processed_path)

    if not os.path.exists(processed_train_path) or not os.path.exists(processed_test_path):
        train_data = DataClass(folder_name=os.path.join(split_path,"train/"))
        test_data = DataClass(folder_name=os.path.join(split_path,"test/"))
        train_data.read_data()
        test_data.read_data()
        train_data.augment_images(train_aug_count)
        test_data.augment_images(test_aug_count)
        train_data.write_data(processed_train_path)
        train_data.write_data(processed_test_path)
