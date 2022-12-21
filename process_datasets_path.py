import os

from utils.utils import get_classes

def txt_data(data_path, classes_name, mode='train'):
    
    if mode == 'train':
        with open('./train_cls.txt', 'a', encoding='utf-8') as f:
            data_classes = os.listdir(data_path)
            for i in range(len(data_classes)):
                img_paths = os.listdir(data_path+'/'+data_classes[i])
                for j in range(len(img_paths)):
                    f.write(str(classes_name.index(data_classes[i]))+';'+data_path+'/'+data_classes[i]+'/'+img_paths[j])
                    f.write('\n')
    else:
        with open('./valid_cls.txt', 'a', encoding='utf-8') as f:
            data_classes = os.listdir(data_path)
            for i in range(len(data_classes)):
                img_paths = os.listdir(data_path+'/'+data_classes[i])
                for j in range(len(img_paths)):
                    f.write(str(classes_name.index(data_classes[i]))+';'+data_path+'/'+data_classes[i]+'/'+img_paths[j])
                    f.write('\n')
    

if __name__ == '__main__':
    train_data_path = './data/train'
    valid_data_path = './data/test'
    classes_name = get_classes('./classes.txt')
   
    txt_data(train_data_path, classes_name, mode='train')
    txt_data(train_data_path, classes_name, mode='valid')
    print("Finished")
    
    