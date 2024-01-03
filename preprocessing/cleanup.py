import cv2 as cv
import os

# TODO: Rename function and class names
class Cleanup:
    images = []
    classes = []

    folder_name = None
    width = 255
    height = 255
    kernel_size = (5,5)
    threshold_value = 127
    max_threshold = 255
    
    def __init__(self, classes, folder_name, width=255, height=255, kernel_size=(5,5), threshold_value=127, max_threshold=255):
        self.classes = classes
        self.folder_name = folder_name
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.threshold_value = threshold_value
        self.max_threshold = max_threshold

    def _read_images_from_folder(self):
        print('reading images...')
        images = []
        for classname in classes:
            for filename in os.listdir(os.path.join(self.folder_name, classname)):
                img = cv.imread(os.path.join(self.folder_name, classname, filename))
                if img is not None:
                    print(img)
                    images.append(img)
        print('done!')
        return images
    
    def _resize_images(self, images):
        print('resizing images...')
        resized_images = []
        for img in images:
            resized_image = cv.resize(img, (self.width, self.height))
            print(resized_images)
            resized_images.append(resized_image)
        print('done!')
        return resized_images
    
    def _convert_to_grayscale(self, images):
        print('converting to grayscale...')
        grayscale_images = []
        for img in images:
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            print(gray_img)
            grayscale_images.append(gray_img)
        print('done!')
        return grayscale_images
    
    def _apply_filter(self, images):
        print('applying gaussian blur...')
        filtered_images = []
        for img in images:
            filtered_img = cv.GaussianBlur(img, self.kernel_size, 0)
            print(filtered_img)
            filtered_images.append(filtered_img)
        print('done!')
        return filtered_images
    
    def _image_segmentation(self, images):
        segmented_images = []
        for img in images:
            _, segmented_img = cv.threshold(
                img,
                self.threshold_value,
                self.max_threshold,
                cv.THRESH_BINARY)
            segmented_images.append(segmented_img)
        return segmented_images
    
    def process_images(self, is_resize=True, convert_to_grayscale=True, apply_filter=True, segment_images=True):
        self.images = self._read_images_from_folder()
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
        images = self.images[top:bottom]
        for image, i in enumerate(images):
            cv.imshow(f"Segmented Image {i + 1}", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

# classes = ['Bud Root Dropping', 'Bud Rot', 'Gray Leaf Spot', 'Leaf Rot', 'Stem Bleeding']
classes = ['Bud Root Dropping'] 
c = Cleanup(folder_name='datasets/Coconut Tree Disease Dataset/', classes=classes)
c.process_images()
c.show_images(0,2)
