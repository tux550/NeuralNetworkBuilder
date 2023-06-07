import os

def paths_dict(dataset_path, wav_limit=None):
    # Init emotions dict
    actors_dict = {}
    # Create dict of emotions dict with filepaths
    for a in os.listdir(dataset_path):
        # Get list of wav paths
        imgpath_ls = [dataset_path+"/"+a+"/"+filename for filename in os.listdir(dataset_path+"/"+a)]
        # Save list of wav paths
        if wav_limit:
            actors_dict[a] = imgpath_ls[:wav_limit]
        else:
            actors_dict[a] = imgpath_ls
    # Return
    return actors_dict
