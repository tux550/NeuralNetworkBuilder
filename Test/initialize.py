import numpy as np
from load.fv import wav2fv
from load.utils import paths_dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DATASET_PATH_SONG = "raw_dataset/sound_dataset/Song"
DATASET_PATH_SPEECH = "raw_dataset/sound_dataset/Speech"
DATASET_PATH_LS = [DATASET_PATH_SONG,DATASET_PATH_SPEECH,] 

def load_wavs(wav_limit = None, fv_size=128):
    # Init X, Y dataset
    raw_X = []
    raw_Y = []
    # Current Label
    curr_label = 0
    actor_label = {}
    # Load dataset
    for dataset_path in DATASET_PATH_LS:
        # Load wavpaths
        actors_dict = paths_dict(dataset_path,wav_limit=wav_limit)    
        # To feature vectors
        for actor in actors_dict:
            print(f"Loading:{dataset_path}-{actor}")
            wavpath_ls = actors_dict[actor]
            for wav_path in wavpath_ls:
                fv = wav2fv(wav_path,fv_size=fv_size)
                raw_X.append(fv)
                if actor in actor_label:
                    raw_Y.append(actor_label[actor])
                else:
                    raw_Y.append(curr_label)
                    actor_label[actor] = curr_label
                    curr_label += 1

    # Dataset to np arrays
    print("Dataset to np")

    raw_Y = [
        [y==i for i in range(len(actor_label.keys()))]
        for y
        in raw_Y
    ]

    X = np.array(raw_X)
    Y = np.array(raw_Y)
    # Return
    return X,Y

def main():
    print("Loading Wavs ...")
    X,Y = load_wavs(fv_size=32) #wav_limit=2)
    print("X shape",X.shape)
    print("Y shape",Y.shape)

    print("MinMax Scaler")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    print("TrainTest Split")
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
    print("X_train shape",X_train.shape)
    print("Y_train shape",Y_train.shape)
    print("X_test shape",X_test.shape)
    print("Y_test shape",Y_test.shape)


    print("Saving to Outputfile")
    np.savetxt("dataset/x_train.csv",X_train,fmt="%.20f")
    np.savetxt("dataset/y_train.csv",Y_train,fmt="%d")
    np.savetxt("dataset/x_test.csv",X_test,fmt="%.20f")
    np.savetxt("dataset/y_test.csv",Y_test,fmt="%d")

if __name__ == "__main__":
    main()



