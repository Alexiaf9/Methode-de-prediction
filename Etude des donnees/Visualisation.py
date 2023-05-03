import pandas as pd
import pyedflib
import numpy as np
import os
from typing import List, Tuple, Optional

class Visualisation:
    """
    La classe Visualisation est une classe pour visualiser les données d'un patient.

    Attributs :
    - patient (str) : le nom du patient
    - description (pd.DataFrame) : un DataFrame contenant des informations sur les données du patient.
    - donnees (pd.DataFrame) : un DataFrame contenant les données du patient
    - informations (pd.DataFrame) : un DataFrame contenant des informations sur les données du patient.
    - to_concat (list) : une liste de DataFrames à concaténer pour la visualisation.
    - noms_fin (list) : une liste de noms pour les DataFrames concaténés.

    Méthodes :
    - chargement(patient, description) : charge les données et les informations du patient dans des DataFrames.
    - selection(informations, D) : sélectionne les DataFrames à concaténer en fonction de la durée.
    """

    def __init__(self, patient: str, description: pd.DataFrame, D: int) -> None:
            """
            Initialise l'objet Visualisation.

            Args :
            - patient (str) : Le nom du patient.
            - description (pd.DataFrame) : Un DataFrame contenant des informations sur les données du patient.
            - D (int) : La durée minimale pour qu'un DataFrame soit inclus dans la visualisation.
            """
            
            # Vérifie si le nom du patient est une chaîne de caractères
            if not isinstance(patient, str):
                raise TypeError("patient should be a string")
            
            # Vérifie si le DataFrame de description contient les colonnes requises
            if not isinstance(description, pd.DataFrame) or 'Enregistrement' not in description.columns or 'Duree D' not in description.columns:
                raise TypeError("description should be a DataFrame with 'Enregistrement' and 'Duree D' columns")
            
            # Charge les données et les informations du patient dans des DataFrames à l'aide de la méthode chargement()
            self.donnees, self.informations = self.chargement(patient, description)
            
            # Sélectionne les DataFrames à concaténer et les noms pour les DataFrames concaténés à l'aide de la méthode selection()
            self.to_concat, self.noms_fin = self.selection(self.informations, D)

            # Optimization
            self.patient = patient
            self.description = description
            self.D = D


    def chargement(self, patient: str, description: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Charge les chemins des fichiers .edf pour un patient donné, ainsi que la description de ses enregistrements.

        Args:
            patient (str): Nom du patient.
            description (pd.DataFrame): DataFrame contenant des informations sur les enregistrements du patient.

        Returns:
            Un tuple contenant la liste des chemins de fichiers .edf pour le patient donné, ainsi qu'un DataFrame contenant
            les informations sur les enregistrements correspondant à ce patient.
        """
        # Créer le chemin vers le dossier contenant les données du patient
        chemins_donnees = f'../../Data/{patient}/'

        # Liste des fichiers .edf dans le dossier du patient
        donnees = [os.path.join(chemins_donnees, f) for f in os.listdir(chemins_donnees) if f.endswith('.edf')]

        # Filtrer la description pour ne garder que les enregistrements du patient donné
        df_patient = description[description['Enregistrement'].str.startswith(patient)]

        return donnees, df_patient


    def selection(self, df: pd.DataFrame, D: int) -> Tuple[List[List[str]], List[str]]:
            """
            Sélectionne les enregistrements et les enregistrements associés à concaténer en fonction de leur durée.

            Args:
                df (pd.DataFrame): Un DataFrame contenant des informations sur les enregistrements.
                D (int): La durée minimale pour qu'un enregistrement soit inclus dans la visualisation.

            Returns:
                Un tuple contenant:
                - name_enr_list (List[List[str]]): Une liste de listes d'enregistrements à concaténer.
                - name_fin_list (List[str]): Une liste de noms pour les enregistrements concaténés.
            """
            # Initialisation des listes pour les noms d'enregistrements à concaténer et pour les noms finaux des enregistrements
            name_enr_list=[]
            name_fin_list=[]

            # Définition des liste des noms des enregistrements ayant la durée D recherchée et des index associés
            enr_list=df[df['Duree D']==D].Enregistrement
            enr_index=df[df.Enregistrement.isin(enr_list)].index.values

            # Boucle sur les index pour chaque enregistrement
            for idx in enr_index:
                # Nom de l'enregistrement contenant la crise
                name_enr=[df['Enregistrement'].loc[idx]]
                name_fin=df['Enregistrement'].loc[idx]
                iter = 1
                # On regarde l'enregistrement précédent
                row = df.loc[idx-iter]
                # Si l'enregistrement de crise a une durée d'avant crise plus courte que la durée D, on cherche les enregistrement d'avant
                while df.loc[idx]['D manquant']>0 and row['Duree']==row['A concatener']:
                    # On ajoute les noms des enregistrements précédents
                    name_enr.append(row.Enregistrement)
                    iter+=1
                    row = df.loc[idx-iter]
                # On inverse la liste des noms d'enregistrements pour qu'ils soient dans l'ordre de la concaténation
                name_enr.reverse()
                # On créer les listes globales
                name_enr_list.append(name_enr)
                name_fin_list.append(name_fin)
            return name_enr_list, name_fin_list
    
    def lecture(self, liste: List[str], donnees: List[str]) -> Tuple[List[pd.DataFrame], Optional[float]]:
        """
        Cette fonction prend en entrée une liste de noms de fichiers "liste" et une liste de chemins de fichiers "donnees".
        Elle retourne une liste de DataFrame pour chaque fichier EDF qui a un nom de fichier correspondant dans "liste",
        et la fréquence d'échantillonnage des signaux. Si aucun fichier EDF ne correspond à la liste donnée, la fonction 
        retourne une liste vide.

        Args:
            liste (List[str]): Une liste de noms de fichiers EDF à extraire
            donnees (List[str]): Une liste de chemins de fichiers EDF existants

        Returns:
            Tuple[List[pd.DataFrame], Optional[float]]: Une liste de DataFrame pour chaque fichier EDF qui a un nom de fichier
            correspondant dans "liste", et la fréquence d'échantillonnage des signaux. Si aucun fichier EDF ne correspond à 
            la liste donnée, la fonction retourne une liste vide. Si les fichiers EDF sont vides ou ne contiennent pas les mêmes 
            fréquences d'échantillonnage, la fonction retourne None pour la fréquence d'échantillonnage.

        Raises:
            TypeError: si "liste" n'est pas une liste de chaînes de caractères, ou si "donnees" n'est pas une liste de chemins 
            de fichiers existants.

        """
        # Vérifie que liste est une liste de chaînes de caractères
        if not isinstance(liste, list) or not all(isinstance(item, str) for item in liste):
            raise TypeError("liste should be a list of strings")
        # Vérifie que donnees est une liste de chemins de fichiers existants
        if not isinstance(donnees, list) or not all(os.path.isfile(filepath) for filepath in donnees):
            raise TypeError("donnees should be a list of file paths")
        # Si donnees est vide, retourne une liste vide
        if not donnees:
            return []
        # Liste des DataFrames pour chaque fichier EDF de la liste
        DF = []
        # Parcourt les fichiers EDF et les traite s'ils sont dans la liste donnée
        sample_freq = None
        for filepath in donnees:
            filename = os.path.basename(filepath)
            # Si le nom de fichier est dans la liste
            if filename in liste:
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
        # Retourne la liste de DataFrames et la fréquence d'échantillonnage
        return DF, sample_freq
    
    def formattage(self, liste: List[str], donnees: List[str], informations: pd.DataFrame) -> Tuple[List[pd.DataFrame], Optional[float]]:
        """Formate les données EEG lues à partir des fichiers EDF pour l'entraînement d'un modèle.

        Args:
            liste (list): Une liste de chaînes de caractères contenant les noms des fichiers EDF à utiliser.
            donnees (list): Une liste de chemins de fichiers existants contenant les données EDF.
            informations (pandas.DataFrame): Un DataFrame contenant les informations de chaque fichier EDF.

        Returns:
            list: Une liste de DataFrames formatés contenant les données EEG et les étiquettes.
            float: La fréquence d'échantillonnage des signaux EEG.

        Raises:
            TypeError: Si les arguments ne sont pas du bon type.

        """
        # Initialise la liste de DataFrames formatés
        DF_format = []
        # Lecture des fichiers EDF
        DF, sample_freq = self.lecture(liste, donnees)
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
        # Retourne la liste de DataFrames formatés et la fréquence d'échantillonnage
        return DF_format, sample_freq
    
    def concatenation(self, formatted_dfs: List[pd.DataFrame], informations: pd.DataFrame, sample_freq: float) -> pd.DataFrame:
        """
        Concatène des enregistrements d'EEG préalablement formatés et nettoyés.

        Parameters
        ----------
        formatted_dfs : list of pandas dataframes
            Une liste de dataframes pandas, chacun correspondant à un enregistrement EEG nettoyé et formaté.

        informations : pandas dataframe
            Un dataframe pandas contenant les informations pertinentes sur chaque enregistrement EEG.

        sample_freq : int
            La fréquence d'échantillonnage utilisée lors de l'enregistrement EEG.

        Returns
        -------
        df_concat : pandas dataframe
            Un dataframe pandas contenant les données EEG concaténées pour tous les enregistrements dans formatted_dfs.
        """
        # Valider les entrées
        if not isinstance(formatted_dfs, list):
            raise TypeError("formatted_dfs doit être une liste de dataframes pandas")
        if not isinstance(informations, pd.DataFrame):
            raise TypeError("informations doit être un dataframe pandas")

        if len(formatted_dfs) == 1:
            # S'il n'y a qu'un seul enregistrement, sélectionner la plage horaire pertinente
            df = formatted_dfs[0]
            info = informations[informations['Enregistrement'] == df['Nom'][0]]
            time_cut = - info['D manquant'].values[0] + info['Debut sec'].values[0]
            df_concat = df[df['time']>time_cut]
            time_cut_crise = info['Fin crise'].values[0] + info['Debut sec'].values[0]
            df_concat= df_concat[df_concat['time']<= time_cut_crise]
        else:
            # S'il y a plusieurs enregistrements, les concaténer et ajouter des lignes NaN si nécessaire
            df_list = []
            for index, df in enumerate(formatted_dfs):
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
                    if index == len(formatted_dfs) - 1:
                        # Pour le dernier enregistrement, sélectionner la plage horaire pertinente
                        time_cut = info['Fin crise'].values[0] + info['Debut sec'].values[0]
                        df = df[df['time']<= time_cut]
                df_list.append(df)
            df_concat = pd.concat(df_list)

        return df_concat
    
    def telechargement(self, patient: str):
        """
        Télécharge les données formatées et concaténées pour un patient donné.

        Args:
            patient (str): Le nom du patient.

        Returns:
            None

        Raises:
            Aucune exception.

        """
        # Parcourir la liste self.to_concat et formater et concaténer les données de chaque enregistrement
        for index, liste in enumerate(self.to_concat):
            # Formater les données
            DF_format, sample_freq = self.formattage(liste, self.donnees, self.informations)
            # Concaténer les données formatées
            df = self.concatenation(DF_format, self.informations, sample_freq)
            # Renommer le fichier .edf en .csv pour enregistrer les données concaténées
            nom = self.noms_fin[index].replace('.edf', '.csv')
            # Enregistrer les données concaténées dans le dossier correspondant au patient
            path = f"../../Data/{patient}/{nom}"
            df.to_csv(path, index=False)
        return