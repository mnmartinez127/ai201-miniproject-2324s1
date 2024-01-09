import cv2,os,random
import numpy as np
import albumentations as A
from imblearn.over_sampling import SMOTE,RandomOverSampler,KMeansSMOTE

#TODO: Add image convolutions
#TODO: Move SMOTE and augmentation (in that order) to preprocessing step
#TODO: Replace images and image_paths dictionaries with 3 lists or a pandas dataframe (not urgent)
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
        self.images = {}
        self.image_paths = {}
        self.data_size = 0

    def _get_classes(self,classes=[]):
        classlist = sorted([classname for classname in os.listdir(self.folder_name)
                            if os.path.isdir(os.path.join(self.folder_name,classname))])
        return classlist if not classes else classlist.intersection(classes)

    def read_data(self, limit=0):
        #reset data arrays
        self.images,self.image_paths = {},{}
        self.data_size = 0
        ctr = 0
        #Read up to <limit> number of images from each class
        print('reading images...')
        for classname in self.classes:
            self.images[classname], self.image_paths[classname] = [],[]
            filenames = os.listdir(os.path.join(self.folder_name, classname))
            if limit > 0:
                filenames = filenames[:limit]
            for filename in filenames:
                file_path = os.path.join(self.folder_name, classname, filename)
                img = cv2.imread(file_path)
                ctr += 1
                print(f'Reading file {ctr}: {file_path}')
                if img is not None:
                    self.images[classname].append(img)
                    self.image_paths[classname].append(file_path)
        print('done!')
        return self.images
    
    def _k_means(self,img):
        Z = np.float32(img.reshape((-1,3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2

    def smote_data(self,X=None,y=None,random_state=None):
        if random_state is not None:
            self.randomizer = np.random.default_rng(random_state)
        if X is None or y is None:
            X,y = self.get_dataset()
        X_shape = X.shape
        X = X.reshape(X.shape[0],-1)#Flatten image matrix before resizing
        sm = SMOTE(random_state=self.randomizer.integers(2**32-1))
        X,y = sm.fit_resample(X,y)
        X = X.reshape(X.shape[0],X_shape[1],X_shape[2],X_shape[3])#Un-flatten image matrix
        return X,y

    def augment_image(self,transform,image,filepath,count=10):
        new_images,new_filepaths = [],[]
        for c in range(1,count+1):
            new_filepaths.append(f'_{c}'.join(os.path.splitext(filepath)))
            new_images.append(transform(image=image)['image'])
        return new_images,new_filepaths

    def augment_images(self,count=10,limit=0):
        if len(self.images) == 0: #load data if not yet loaded or processed
            self.images = self.read_data(limit)
        #Pipeline used in augmentations
        transform = A.Compose([
        #flip and randomly scale and rotate the image
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.625, scale_limit=0.1, rotate_limit=60, interpolation = 3, border_mode = 3, p=0.8),
        ])
        ctr = 0
        print("Augmenting...")
        for classname in self.images.keys():
            new_images,new_image_paths = [],[]
            for i in range(len(self.images[classname])):
                new_images_i,new_image_paths_i = self.augment_image(transform,self.images[classname][i],self.image_paths[classname][i],count)
                new_images.extend(new_images_i)
                new_image_paths.extend(new_image_paths_i)
                ctr += 1
                print(f'Augmented file {ctr}: {self.image_paths[classname][i]}')
            #add new images after processing all images in a class
            self.images[classname].extend(new_images)
            self.image_paths[classname].extend(new_image_paths)
        print("Data augmentation complete!")

    def process_images(self, resize=True, convert_to_grayscale=False,
                        apply_filter=False, segment_images=False, k_means = False,
                          display_images=False, limit=0):
        ctr = 0
        if len(self.images) == 0: #load data if not yet loaded or processed
            self.images = self.read_data(limit)
        processed_images = {}
        for classname in self.images.keys():
            processed_images[classname] = []
            for i in range(len(self.images[classname])):
                #process image and generate segmentation masks separately
                processed_img = np.copy(self.images[classname][i])
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
                    cv2.imshow(f"Original Image", self.images[classname][i])
                    cv2.imshow(f"Gray Image", gray_img)
                    cv2.imshow(f"Blurred Image", blur_img)
                    cv2.imshow(f"Segmentation Mask", segmented_mask)
                    cv2.imshow(f"Segmented Image", segmented_img)
                    cv2.imshow(f"Processed Image", processed_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                processed_images[classname].append(processed_img)
                ctr += 1
                print(f'Processed file {ctr}: {self.image_paths[classname][i]}')
        self.images = processed_images
        print('done!')

    def get_dataset(self,limit=0):
        if len(self.images) == 0: #load data if not yet loaded or processed
            self.images = self.read_data(limit)
        X,y = [],[]
        for classname in self.classes:
            for img in self.images[classname]:
                X.append(img)
                y.append(classname)
        return np.array(X),np.array(y)

    def show_images(self, start, stop=None):
        if not stop: start,stop = 0,start
        for cl in self.classes:
            images = self.images[cl][start:stop]
            for i,image in enumerate(images):
                cv2.imshow(f"Segmented Image {i + 1}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def write_images(self, processed_path='datasets/processed'):
        print(f'Creating processed folder...')
        os.makedirs(processed_path,exist_ok=True)
        ctr = 0
        for classname in self.images:
            dir_path = os.path.join(processed_path, classname)
            try:
                print(f'Creating folder {classname}...')
                os.mkdir(dir_path)
            except FileExistsError:
                print('Folder already exists!')
                pass
            for i, img in enumerate(self.images[classname]):
                file_path = os.path.join(dir_path, os.path.split(self.image_paths[classname][i])[1])
                ctr += 1
                print(f'Writing file {ctr}: {self.image_paths[classname][i]}->{file_path}')
                cv2.imwrite(file_path, img)
                self.image_paths[classname][i] = file_path

if __name__ == "__main__":
    data = DataClass(folder_name='datasets/Coconut Tree Disease Dataset/')
    data.read_data(10)
    data.augment_images(2)
    #preprocess all data
    data.process_images(convert_to_grayscale=False,apply_filter=False,k_means=False,segment_images=True)
    print(f"Preprocessing complete!")
    data.write_images()
    print(f"Writing complete!")
    #data.show_images(0,3)
