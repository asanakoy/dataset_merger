import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import deepdish as dd
import sys
import os


wiki_img_folder_path = '/export/home/asanakoy/workspace/wikiart/images/'
moma_img_folder_path = '/export/home/asanakoy/workspace/moma/images/1_filtered/'


def false_neg_pairs_visualization(first_img_folder_path, second_img_folder_path, sub_visual_folder, false_neg_pairs_name, false_neg_artists):
    """
    visualization of neg. pairs
    Args:
        first_img_folder_path: img path for the fist art gallery
        second_img_folder_path: img path for the second art gallery
        sub_visual_folder: path for save data
        false_neg_pairs_name: false neg pairs id
        false_neg_artists:false neg artist names

    """
    num_iters = len(false_neg_pairs_name)
    for i in range(num_iters):
        name_for_save = `i` + '_pair' + '.jpg'
        full_path = os.path.join(sub_visual_folder, name_for_save)
        wiki_ele_name = false_neg_pairs_name[i][0]
        moma_ele_name = false_neg_pairs_name[i][1]
        wiki_ele = os.path.join(first_img_folder_path, wiki_ele_name)
        moma_ele = os.path.join(second_img_folder_path, moma_ele_name)
        # plot them and save
        img_wiki = mpimg.imread(wiki_ele)
        img_moma = mpimg.imread(moma_ele)
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        implot = plt.imshow(img_wiki)
        a.set_title(wiki_ele_name)
        a.set_xlabel(false_neg_artists[i][0])
        a = fig.add_subplot(1,2,2)
        implot = plt.imshow(img_moma)
        a.set_title(moma_ele_name)
        a.set_xlabel(false_neg_artists[i][1])
        plt.savefig(full_path)
        plt.close()
        sys.stdout.write("\r%d/%d" % (i+1, num_iters))
        sys.stdout.flush()


if __name__ == '__main__':
    # test
    visualization_folder_path = '/export/home/jli/workspace/data_after_run/visualization_wiki_moma/'
    false_neg_artists = dd.io.load('/export/home/jli/workspace/data_after_run/false_pos_pairs_artists_wiki_moma.h5')
    false_neg_pairs_name = dd.io.load('/export/home/jli/workspace/data_after_run/false_pos_pairs_path_wiki_moma.h5')
    false_neg_pairs_visualization(wiki_img_folder_path, moma_img_folder_path, visualization_folder_path, false_neg_pairs_name, false_neg_artists)
    first_img_folder_path = dd.io.load('/export/home/jli/workspace/readable_code_data/path_combine_wiki_moma.h5')
    second_img_folder_path = dd.io.load('/export/home/jli/workspace/readable_code_data/img_path_list_rijks.h5')
    false_neg_pairs_name = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/false_pos_pairs_path_wikimoma_rijks.h5')
    false_neg_artists = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/false_pos_pairs_artists_wikimoma_rijks.h5')
    #false_neg_pairs_visualization(first_img_folder_path, second_img_folder_path, visualization_folder_path, false_neg_pairs_name, false_neg_artists)