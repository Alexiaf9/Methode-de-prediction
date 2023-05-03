import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

class Affichage:

    def __init__(self, patient):
        self.all = glob.glob('../../Data/' + patient + '/*.csv')
    
    def aff_eeg(self, df, telechargement, im_path):
        change_points = np.where(np.abs(np.diff(df['label'])) == 1)[0]
        xlim = [df.index[0], df.index[-1]]
        ylim = [df.min().min(), df.max().max()]
        fig, axs = plt.subplots(df.shape[1]-1, 1, figsize=(14, 6*(df.shape[1]-1)))
        for i, col in enumerate(df.columns[:-1]):
            axs[i].plot(df.index, df[col])
            axs[i].set_title(col)
            axs[i].set_ylim(ylim)
            axs[i].set_xlim(xlim)
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Amplitude')
        for point in change_points:
            for i in range(df.shape[1]-1):
                axs[i].axvline(x=df.index[point], color='r')
        plt.tight_layout()
        if telechargement:
            fig.savefig(im_path)
        else:
            plt.show()
        plt.close(fig)
        return

    def aff_all_eeg(self):
        for path in self.all:
            df = pd.read_csv(path, index_col = 'time')
            df = df.drop(['Ind', 'Nom'], axis = 1)
            labels = list(df['label'].unique())
            filename = os.path.basename(path).replace('.csv','.png')
            im_path = f"../../Data/Images/Images_brutes/{filename}"
            self.aff_eeg(df, True, im_path)
        return