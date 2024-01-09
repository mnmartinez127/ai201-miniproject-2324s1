import cv2 as cv
import os

# TODO: Rename function and class names
class Cleanup:
    images = {}
    classes = []

    folder_name = None
    width = 255
    height = 255
    kernel_size = (5,5)
    threshold_value = 127
    max_threshold = 255
    
    def __init__(self, classes, folder_name, width=255, height=255, kernel_size=(5,5), threshold_value=127, max_threshold=255):
        self.folder_name = folder_name
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.threshold_value = threshold_value
        self.max_threshold = max_threshold
        self.classes = classes


    def _read_images_from_folder(self, limit=-1):
        print('reading images...')
        for classname in self.classes:
            self.images[classname] = []
            for filename in os.listdir(os.path.join(self.folder_name, classname))[:limit]:
                img = cv.imread(os.path.join(self.folder_name, classname, filename))
                if img is not None:
                    #print(img)
                    self.images[classname].append(img)
        print('done!')
        return self.images
    
    def _resize_images(self, images):
        """ Note: The parameter `images` is a dictionary that splits the
        different image parameters into their respective classes.
        """
        print('resizing images...')
        resized_images = {}
        for cl in images.keys():
            resized_images[cl] = []
            for img in images[cl]:
                resized_image = cv.resize(img, (self.width, self.height))
                #print(resized_images)
                resized_images[cl].append(resized_image)
        print('done!')
        return resized_images
    
    def _convert_to_grayscale(self, images):
        """ Note: The parameter `images` is a dictionary that splits the
        different image parameters into their respective classes.
        """
        print('converting to grayscale...')
        grayscale_images = {}
        for cl in images.keys():
            grayscale_images[cl] = []
            for img in images[cl]:
                gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                #print(gray_img)
                grayscale_images[cl].append(gray_img)
        print('done!')
        return grayscale_images
    
    def _apply_filter(self, images):
        """ Note: The parameter `images` is a dictionary that splits the
        different image parameters into their respective classes.
        """
        print('applying gaussian blur...')
        filtered_images = {}
        for cl in images.keys():
            filtered_images[cl] = []
            for img in images[cl]:
                filtered_img = cv.GaussianBlur(img, self.kernel_size, 0)
                #print(filtered_img)
                filtered_images[cl].append(filtered_img)
        print('done!')
        return filtered_images
    
    def _image_segmentation(self, images):
        """ Note: The parameter `images` is a dictionary that splits the
        different image parameters into their respective classes.
        """
        print('applying image segmentation...')
        segmented_images = {}
        for cl in images.keys():
            segmented_images[cl] = []
            for img in images[cl]:
                _, segmented_mask = cv.threshold(
                    img,
                    self.threshold_value,
                    self.max_threshold,
                    cv.THRESH_BINARY)
                segmented_img = cv.bitwise_and(img,img,mask=segmented_mask)
                segmented_images[cl].append(segmented_img)
        return segmented_images
    
    def process_images(self, is_resize=True, convert_to_grayscale=True, apply_filter=True, segment_images=True, limit=10):
        self.images = self._read_images_from_folder(limit)
        if is_resize:
            self.images = self._resize_images(self.images)
        if convert_to_grayscale:
            self.images = self._convert_to_grayscale(self.images)
        if apply_filter:
            self.images = self._apply_filter(self.images)
        if segment_images:
            self.images = self._image_segmentation(self.images)
                
    def show_images(self, top, bottom=None):
        """Note: top and bottom are slicing offsets"""
        if not bottom:
            bottom = len(self.images)
        for cl in self.classes:
            images = self.images[cl][top:bottom]
            for i,image in enumerate(images):
                cv.imshow(f"Segmented Image {i + 1}", image)
                cv.waitKey(0)
                cv.destroyAllWindows()

    def write_images(self, processed_path='preprocessing/datasets/processed'):
        try:
            print(f'Creating processed folder...')
            os.mkdir(processed_path)
        except FileExistsError:
            print('processed folder already exists')
            pass
        for cl in self.images:
            dir_path = os.path.join(processed_path, cl)
            try:
                print(f'Creating folder {cl}...')
                os.mkdir(dir_path)
            except FileExistsError:
                print('Folder already exists!')
                pass
            for i, img in enumerate(self.images[cl]):
                file_path = os.path.join(dir_path, (str(i) + '.png'))
                print(f'Writing file: {file_path}')
                cv.imwrite(file_path, img)


classes = ['Bud Root Dropping', 'Bud Rot', 'Gray Leaf Spot', 'Leaf Rot', 'Stem Bleeding']
c = Cleanup(folder_name='preprocessing/datasets/Coconut Tree Disease Dataset/', classes=classes)
c.process_images()
c.show_images(0,2)
c.write_images()
