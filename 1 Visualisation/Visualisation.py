import pandas as pd
import pyedflib
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


def initialisation(description_path: str, patient_nb: int, patients_path: str) -> Tuple[List[str], pd.DataFrame, str]:
    """
    Initialise les données en chargeant la description, en sélectionnant un patient spécifique,
    en créant le chemin vers le dossier contenant les données du patient, et en filtrant la
    description pour ne garder que les enregistrements du patient donné.

    Args:
        description_path (str): Le chemin vers le fichier CSV contenant la description des données.
        patient_nb (int): Le numéro du patient souhaité.
        patients_path (str): Le chemin vers le dossier contenant les données des patients.

    Returns:
        Tuple[List[str], pd.DataFrame, str]: Un tuple contenant :
            - Une liste des chemins vers les fichiers .edf du patient.
            - Un DataFrame contenant les informations filtrées pour le patient donné.
            - Le chemin vers le dossier contenant les données du patient.

    Raises:
        FileNotFoundError: Si le fichier de description n'existe pas.
        AssertionError: Si la description n'est pas un DataFrame avec les colonnes requises.
        TypeError: Si le numéro du patient n'est pas valide ou si le nom du patient n'est pas une chaîne de caractères.
    """
    
    # Chargement de la description des données
    try:
        description = pd.read_csv(description_path, delimiter=';')
    except FileNotFoundError:
        print(f"{description_path} n'existe pas")
        exit()

    # Vérifie si le DataFrame de description contient les colonnes requises
    assert isinstance(description, pd.DataFrame) and 'Enregistrement' in description.columns and 'Duree D' in description.columns, "description should be a DataFrame with 'Enregistrement' and 'Duree D' columns"

    # Selection du patient voulu
    patients = [f'chb{i:02d}' for i in range(1, 24)]
    if patient_nb < 1 or patient_nb > len(patients):
        print(f"Patient {patient_nb} n'existe pas")
        exit()
    patient = patients[patient_nb - 1]

    # Vérifie si le nom du patient est une chaîne de caractères
    if not isinstance(patient, str):
        raise TypeError("patient should be a string")

    # Vérifie si le DataFrame de description contient les colonnes requises
    if not isinstance(description, pd.DataFrame) or 'Enregistrement' not in description.columns or 'Duree D' not in description.columns:
        raise TypeError("description should be a DataFrame with 'Enregistrement' and 'Duree D' columns")

    # Créer le chemin vers le dossier contenant les données du patient
    chemins_donnees = f'{patients_path}/{patient}/'

    # Liste des fichiers .edf dans le dossier du patient
    donnees = [os.path.join(chemins_donnees, f) for f in os.listdir(chemins_donnees) if f.endswith('.edf')]

    # Filtrer la description pour ne garder que les enregistrements du patient donné
    informations = description[description['Enregistrement'].str.startswith(patient)]

    return donnees, informations, chemins_donnees

def selection(informations: pd.DataFrame, Duree_D: int) -> Tuple[List[List[str]], List[str]]:
    """
    Sélectionne les enregistrements ayant la durée spécifiée et construit les noms des enregistrements à concaténer.

    Args:
        informations (pd.DataFrame): Un DataFrame contenant les informations des enregistrements.
        Duree_D (int): La durée recherchée pour les enregistrements.

    Returns:
        Tuple[List[List[str]], List[str]]: Un tuple contenant :
            - Une liste de listes de noms d'enregistrements à concaténer dans l'ordre.
            - Une liste des noms finaux des enregistrements.

    """

    # Initialisation des listes pour les noms d'enregistrements à concaténer et pour les noms finaux des enregistrements
    to_concat = []
    noms_fin = []

    # Définition des liste des noms des enregistrements ayant la durée D recherchée et des index associés
    enr_list = informations[informations['Duree D']==Duree_D].Enregistrement
    enr_index = informations[informations.Enregistrement.isin(enr_list)].index.values

    # Boucle sur les index pour chaque enregistrement
    for idx in enr_index:
        # Nom de l'enregistrement contenant la crise
        name_enr=[informations['Enregistrement'].loc[idx]]
        name_fin=informations['Enregistrement'].loc[idx]
        iter = 1
        # On regarde l'enregistrement précédent
        row = informations.loc[idx-iter]
        # Si l'enregistrement de crise a une durée d'avant crise plus courte que la durée D, on cherche les enregistrement d'avant
        while informations.loc[idx]['D manquant']>0 and row['Duree']==row['A concatener']:
            # On ajoute les noms des enregistrements précédents
            name_enr.append(row.Enregistrement)
            iter+=1
            row = informations.loc[idx-iter]
            if row['A concatener'] > 0:
                name_enr.append(row.Enregistrement)
        # On inverse la liste des noms d'enregistrements pour qu'ils soient dans l'ordre de la concaténation
        name_enr.reverse()
        # On créer les listes globales
        to_concat.append(name_enr)
        noms_fin.append(name_fin)

    return to_concat, noms_fin
    
def lecture(donnees: List[str], liste_crise: List[str]) -> Tuple[List[pd.DataFrame], Optional[int]]:
    """
    Lit les fichiers EDF dans la liste donnée et crée des DataFrames à partir des données.

    Args:
        donnees (List[str]): Une liste de chemins vers les fichiers EDF.
        liste_crise (List[str]): Une liste des noms de fichiers à traiter.

    Returns:
        Tuple[List[pd.DataFrame], Optional[int]]: Un tuple contenant :
            - Une liste de DataFrames contenant les données des fichiers EDF traités.
            - La fréquence d'échantillonnage commune à tous les fichiers, ou None s'ils ont des fréquences différentes.

    """

    # Liste des DataFrames pour chaque fichier EDF de la liste
    DF = []

    # Parcourt les fichiers EDF et les traite s'ils sont dans la liste donnée
    sample_freq = None
    for filepath in donnees:
        filename = os.path.basename(filepath)
        # Si le nom de fichier est dans la liste
        if filename in liste_crise:
            try:
                # Ouvre le fichier EDF avec pyedflib
                with pyedflib.EdfReader(filepath) as f:
                    # Obtient les noms des canaux et la fréquence d'échantillonnage
                    channel_names = f.getSignalLabels()
                    curr_sample_freq = f.getSampleFrequency(0)
                    # Vérifie que tous les fichiers ont la même fréquence d'échantillonnage
                    if sample_freq is None:
                        sample_freq = curr_sample_freq
                    elif sample_freq != curr_sample_freq:
                        sample_freq = None
                    # Lit les données pour chaque canal et les stocke dans un dictionnaire
                    data = {}
                    for i, channel in enumerate(channel_names):
                        data[channel] = f.readSignal(i)
                    # Crée un DataFrame à partir du dictionnaire et ajoute le nom de fichier en tant que colonne
                    df = pd.DataFrame.from_dict(data)
                    df['Nom'] = filename
                    # Ajoute le DataFrame à la liste
                    DF.append(df)
            # S'il y a une erreur, passe au fichier suivant
            except (AssertionError, ValueError):
                continue
    return DF, sample_freq

def formattage(sample_freq: int, DF: List[pd.DataFrame], informations: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Formate les données des fichiers EDF en ajoutant des colonnes et en modifiant leur structure.

    Args:
        sample_freq (int): La fréquence d'échantillonnage commune à tous les fichiers EDF.
        DF (List[pd.DataFrame]): Une liste de DataFrames contenant les données des fichiers EDF.
        informations (pd.DataFrame): Un DataFrame contenant les informations des enregistrements.

    Returns:
        List[pd.DataFrame]: Une liste de DataFrames formatés avec les colonnes ajoutées et la structure modifiée.

    """

    # Initialise la liste de DataFrames formatés
    DF_format = []

    # Réinitialisation de l'index de informations
    informations = informations.reset_index()

    # Formatage des données de chaque fichier EDF
    for index, df in enumerate(DF):
        # Définition du temps de début et de fin
        t_debut = 0
        t_fin = len(df) / sample_freq
        # Création de la colonne "time"
        df['time'] = np.linspace(t_debut, t_fin, len(df), endpoint=False)
        # Récupération des informations pour le fichier EDF actuel
        information = informations[informations['Enregistrement'] == df['Nom'][0]]
        if information['Nb crises'].values[0] >= 2:
            information = informations.loc[(informations['Duree D'] == 120) & (informations['Enregistrement'] == df['Nom'][0])]
        # Définition du temps de début du fichier EDF actuel
        debut = information['Debut sec'].values[0]
        # Ajout du temps de début au temps de chaque ligne de la colonne "time"
        df['time'] += debut
        # On enlève les colonnes inutiles
        df = df.loc[:, (df != 0).any(axis=0)]
        # Ajout de la colonne "label" initialisée à 0
        df['label'] = 0
        # Ajout de la colonne "Ind" initialisée à l'indice du fichier EDF
        df['Ind'] = index
        # Si le nombre de crises est supérieur à 0
        if information['Nb crises'].values[0] != 0:
            if index == len(DF) - 1:
                # Récupération du temps de début de la première crise
                debut_crise = information['Debut crise'].values[0]
                # Ajout de la valeur 1 dans la colonne "label" pour les lignes où le temps est supérieur ou égal au temps de début de la première crise
                df.loc[df['time'] >= debut_crise + debut, 'label'] = 1
        # Ajout du DataFrame formaté à la liste DF_format
        DF_format.append(df)

    return DF_format

def concatenation(DF_format: List[pd.DataFrame], informations: pd.DataFrame, sample_freq: int) -> pd.DataFrame:
    """
    Concatène les DataFrames formatés en un seul DataFrame, en sélectionnant les plages horaires pertinentes.

    Args:
        DF_format (List[pd.DataFrame]): Une liste de DataFrames formatés.
        informations (pd.DataFrame): Un DataFrame contenant les informations des enregistrements.
        sample_freq (int): La fréquence d'échantillonnage commune à tous les fichiers EDF.

    Returns:
        pd.DataFrame: Un DataFrame contenant la concaténation des données des enregistrements, avec les plages horaires pertinentes.

    Raises:
        TypeError: Si DF_format n'est pas une liste de DataFrames pandas, ou si informations n'est pas un DataFrame pandas.

    """

    # Valider les entrées
    if not isinstance(DF_format, list):
        raise TypeError("formatted_dfs doit être une liste de dataframes pandas")
    if not isinstance(informations, pd.DataFrame):
        raise TypeError("informations doit être un dataframe pandas")

    if len(DF_format) == 1:
        # S'il n'y a qu'un seul enregistrement, sélectionner la plage horaire pertinente
        df = DF_format[0]
        info = informations[informations['Enregistrement'] == df['Nom'][0]]
        time_cut = - info['D manquant'].values[0] + info['Debut sec'].values[0]
        df_concat = df[df['time']>time_cut]
        time_cut_crise = info['Fin crise'].values[0] + info['Debut sec'].values[0]
        df_concat= df_concat[df_concat['time']<= time_cut_crise]
    else:
        # S'il y a plusieurs enregistrements, les concaténer et ajouter des lignes NaN si nécessaire
        df_list = []
        for index, df in enumerate(DF_format):
            info = informations[informations['Enregistrement'] == df['Nom'][0]]
            if index == 0:
                # Pour le premier enregistrement, sélectionner la plage horaire pertinente
                time_cut = info['Fin sec'].values[0] - info['A concatener'].values[0]
                df = df[df['time'] >= time_cut]
                df = df.reset_index(drop=True)
            else:
                # Pour les enregistrements suivants, ajouter des lignes NaN au début
                decalage = info['Decalage'].values[0]
                nan_rows = pd.DataFrame(np.nan, index=np.arange(decalage * sample_freq), columns=df.columns)
                start_time = info['Debut sec'].values[0] - decalage
                end_time = info['Debut sec'].values[0]
                num_samples = len(nan_rows)
                nan_rows['time'] = np.linspace(start_time, end_time, num_samples, endpoint=False)
                nan_rows['label'] = 0
                nan_rows['Ind'] = df['Ind'][0]
                nan_rows['Nom'] = df['Nom'][0]
                df = pd.concat([nan_rows, df])
                if index == len(DF_format) - 1:
                    # Pour le dernier enregistrement, sélectionner la plage horaire pertinente
                    time_cut = info['Fin crise'].values[0] + info['Debut sec'].values[0]
                    df = df[df['time']<= time_cut]
            df_list.append(df)
        df_concat = pd.concat(df_list)
        
    return df_concat

def telechargement(nom_fin: str, chemins_donnees: str, df_concat: pd.DataFrame) -> None:
    """
    Enregistre les données concaténées dans un fichier CSV.

    Args:
        nom_fin (str): Le nom final du fichier .edf concaténé.
        chemins_donnees (str): Le chemin vers le dossier contenant les données du patient.
        df_concat (pd.DataFrame): Le DataFrame contenant les données concaténées.

    Returns:
        None

    """

    # Renommer le fichier .edf en .csv pour enregistrer les données concaténées
    nom = nom_fin.replace('.edf', '.csv')

    # Enregistrer les données concaténées dans le dossier correspondant au patient
    path = f"{chemins_donnees}/{nom}"
    df_concat.to_csv(path, index=False)
    return

def affichage(csv: str, telecharge: bool) -> None:
    """
    Affiche les données contenues dans un fichier CSV sous forme de graphiques.

    Args:
        csv (str): Le chemin vers le fichier CSV contenant les données.
        telecharge (bool): Indique si l'image du graphique doit être téléchargée ou affichée.

    Returns:
        None

    """

    # On charge la crise voulue et on la formatte
    df = pd.read_csv(csv)
    df = df.set_index('time')
    df = df.drop(['Ind', 'Nom'], axis = 1)

    # On récupère l'indice de début de crise
    change_points = np.where(np.abs(np.diff(df['label'])) == 1)[0]

    # On défini les limites des graphiques
    xlim = [df.index[0], df.index[-1]]
    ylim = [df.min().min(), df.max().max()]

    # Affichage
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
    if telecharge:
        # Chemin de téléchargement
        filename = os.path.basename(csv).replace('.csv','.png')
        im_path = f"C:/Users/Alexia/Desktop/Data/Images/Images_brutes/{filename}"
        # Téléchargement
        fig.savefig(im_path)
    else:
        plt.show()
    plt.close()
    return