import librosa
import numpy as np
import pandas as pd
from glob import glob 
from tqdm.notebook import tqdm
import warnings
from datetime import date
import dask
from dask import delayed
from dask.distributed import Client


warnings.filterwarnings("ignore")

class Features:
    
    def __init__(self, 
                 metadata_path="data/raw/metadata/UrbanSound8K.csv", 
                 audio_files_path="data/raw/audio", 
                 save_training_path="data/processed",
                 folds=5, 
                 workers=4):
        """
        Initialize Feature object

        Args:
            metadata_path (str, optional): csv dataset metadata. Defaults to "data/UrbanSound8K/metadata/UrbanSound8K.csv".
            audio_files_path (str, optional): folder with the audio files. Defaults to "data/UrbanSound8K/audio".
            folds (int, optional): first N folds to extract. Defaults to 5.
        """
        self.metadata_path = metadata_path
        self.audio_files_path = audio_files_path
        self.folds = folds
        self.save_training_path = f"{save_training_path}/{date.today().strftime('%m_%d_%Y')}.csv"
        self.workers = workers

    def get_features(self, audio_file):
        """
        Extract features from an audio file using Librosa

        Args:
            audio_file (str): audio file path to load
        """
        def array_map(array):
            return [
                    np.min(array), 
                    np.max(array),
                    np.median(array), 
                    np.mean(array)
                ]
        
        def array_reduce(a, b):
            return a + b
        
        y, sr = librosa.load(audio_file, sr = None)
            
        mfcc = librosa.feature.mfcc(y, sr)
        chroma_stft = librosa.feature.chroma_stft(y, sr)
        rms = librosa.feature.rms(y, sr)
        
        total = np.concatenate((mfcc, chroma_stft, rms), axis=0)
        return np.apply_along_axis(array_map, 1, total).flatten()


    def get_training_dataframe(self):
        """
        Get the dataframe that represent the training dataset. 
        The structure is [class, feature_1, feature_2, ...]
        Dask is used to speed up the computation.
        
        Returns:
            Pandas Dataframe: training dataset
        """
        data = pd.read_csv(self.metadata_path)
        training_data = data[data["fold"] <= self.folds]
        values = training_data[["slice_file_name", "fold", "classID"]].values
        
        @delayed
        def m(x):
            audio_path = f"{self.audio_files_path}/fold{x[1]}/{x[0]}"
            return np.insert(self.get_features(audio_path), 0, int(x[2]))
        
        Client(n_workers = self.workers)
        
        feature_arrays = []
        for i in tqdm(range(len(values))):
            r = m(values[i])
            feature_arrays.append(r)
        
        feature_arrays = dask.compute(*feature_arrays)
        
        columns = ["class"] + [f"f_{i}" for i in range(len(feature_arrays[0]) -1)]
        return pd.DataFrame(feature_arrays, columns=columns)
    
    def save_training_set(self, dataframe):
        dataframe.to_csv(self.save_training_path, index=False)