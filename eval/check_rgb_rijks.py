from scipy.misc import imread
import deepdish as dd
import Image
import os
import sys
import pickle

folder = '/export/home/jli/workspace/art_project/practical/data/'
final_folder = '/export/home/jli/workspace/art_project/practical/data/final/'
rijks_img_folder = '/export/home/jli/workspace/rijks_images/jpg2/'
with open(os.path.join(folder, 'rijks_img_path_list'), 'rb') as fp:
    path_rijks = pickle.load(fp)

count = 0

def convert2RGB(images_list, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print 'converting now!'
    cnt = 1
    for path_to_image in images_list:
        sys.stdout.write("\r%d/%d" % (cnt, len(images_list)))
        sys.stdout.flush()
        cnt += 1
        image = Image.open(path_to_image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        file_to_save = os.path.join(output_dir, os.path.splitext(os.path.basename(path_to_image))[0] + '.jpg')
        print file_to_save

        #if not os.path.exists(file_to_save):
        image.save(file_to_save)
    print 'okay'

save_name = 'not_RGB_rijks_list.h5'
if not os.path.exists(os.path.join(final_folder, save_name)):
    print 'checking now!'
    need_convert_list = []
    for i in range(len(path_rijks)):
        image = imread(path_rijks[i])
        if(len(image.shape) != 3):
            need_convert_list.append(path_rijks[i])
        count += 1
        sys.stdout.write("\r%d/%d" % (i+1, len(path_rijks)))
        sys.stdout.flush()
    # save this list
    dd.io.save(os.path.join(final_folder, save_name), need_convert_list)


not_RGB_list = dd.io.load(os.path.join(final_folder, save_name))
convert2RGB(not_RGB_list, rijks_img_folder)