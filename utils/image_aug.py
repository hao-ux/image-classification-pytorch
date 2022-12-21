from PIL import Image, ImageEnhance
import numpy as np

# ---------------------------------- #
# 1. Cutout数据增强
# ---------------------------------- #
class Cutout(object):
    def __init__(self, n_holes=1, length=112, prob=0.5):
        super(Cutout, self).__init__()
        self.prob = prob
        self.n_holes = n_holes
        self.length = length
        
    def __call__(self, img):
        img = np.array(img)
        if np.random.random() < self.prob:
            h, w = img.shape[:2]
            for _ in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                img[y1:y2, x1:x2] = 0
        return img

# ---------------------------------- #
# 2. 随机图像翻转
# ---------------------------------- #
class RandomHorizontalVerticalFlip(object):
    def __init__(self, prob0=0.3, prob1=0.4):
        super(RandomHorizontalVerticalFlip, self).__init__()
        self.prob0 = prob0
        self.prob1 = prob1
        assert prob0 + prob1 < 1, "Both prob0 and prob1 are wrong settiing, prob0 + prob1 <= 1 (prob0>=0, prob1>=1)"
    
    def __call__(self, img):
        img = np.array(img)
        rand_prob = np.random.random()
        if rand_prob < self.prob0:
            img = img[::-1,:, :]
        elif self.prob0 < rand_prob < self.prob0 + self.prob1:
            img = img[:,::-1, :]
        return img

# ---------------------------------- #
# 3. 调整图片大小
# ---------------------------------- #
class Resize(object):
    def __init__(self, target_size=None):
        super(Resize, self).__init__()
        self.target_size = target_size
        if isinstance(self.target_size, (list, tuple)):
            self.target_size = self.target_size[0]
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        return img
    
# ---------------------------------- #
# 4. 随机旋转图像
# ---------------------------------- #
class RandomRotate(object):
    def __init__(self, prob=0.5):
        super(RandomRotate, self).__init__()
        self.prob = prob
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if np.random.random() < self.prob:
            angle = np.random.randint(-10, 10)
            img = img.rotate(angle)
        
        return img

# ---------------------------------- #
# 5. 随机亮度调整
# ---------------------------------- #
class RandomBright(object):
    def __init__(self, prob=0.5, brightness_delta=0.225):
        super(RandomBright, self).__init__()
        self.prob = prob
        self.brightness_delta = brightness_delta
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if np.random.random() < self.prob:
            delta = np.random.uniform(-self.brightness_delta, self.brightness_delta) + 1
            img = ImageEnhance.Brightness(img).enhance(delta)
        
        return img
    
# ---------------------------------- #
# 6. 随机对比度调整
# ---------------------------------- #
class RandomContrast(object):
    def __init__(self, prob=0.5, contrast_delta=0.5):
        super(RandomContrast, self).__init__()
        self.prob = prob
        self.contrast_delta = contrast_delta
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if np.random.random() < self.prob:
            delta = delta = np.random.uniform(-self.contrast_delta, self.contrast_delta) + 1
            img = ImageEnhance.Contrast(img).enhance(delta)
        
        return img
    
# ---------------------------------- #
# 7. 随机饱和度调整
# ---------------------------------- #
class RandomSaturation(object):
    def __init__(self, prob=0.5, saturation_delta=0.5):
        super(RandomSaturation, self).__init__()
        self.prob = prob
        self.saturation_delta = saturation_delta
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if np.random.random() < self.prob:
            delta = np.random.uniform(-self.saturation_delta, self.saturation_delta) + 1
            img = ImageEnhance.Color(img).enhance(delta)
        
        return img

# ---------------------------------- #
# 8. 随机色度调整
# ---------------------------------- #
class RandomHue(object):
    def __init__(self, prob=0.5, hue_delta=18):
        super(RandomHue, self).__init__()
        self.prob = prob
        self.hue_delta = hue_delta
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if np.random.random() < self.prob:
            delta = np.random.uniform(-self.hue_delta, self.hue_delta)
            img_hsv = np.array(img.convert('HSV'))
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
            img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
        
        return img