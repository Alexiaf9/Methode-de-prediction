# Etape 1 - Etude des données
*Alexia Fournier*
## Introduction
Le projet de fin d'études en question a un objectif clair et ambitieux : développer une méthode pour prédire les crises d'épilepsie à partir de données EEG/iEEG. Pour atteindre ce but, on a choisi la base de données CHBMIT, une source de données de qualité et reconnue dans le domaine de la recherche sur l'épilepsie.

Cependant, avant de pouvoir élaborer une nouvelle méthode, il est crucial de bien comprendre les subtilités de l'ensemble de données CHBMIT. C'est pourquoi la première étape du projet consiste à visualiser et analyser les données. Cette étape permettra aux chercheurs d'avoir une meilleure compréhension des caractéristiques de l'ensemble de données, telles que la qualité des signaux EEG/iEEG et la fréquence des crises d'épilepsie enregistrées.

Une analyse minutieuse des données permettra également d'identifier les caractéristiques les plus significatives à prendre en compte pour la prédiction des crises d'épilepsie. Cette première étape est donc essentielle pour le succès du projet, car elle permettra de développer une méthode de prédiction précise et fiable.

En somme, le projet de fin d'études est un projet de recherche passionnant qui a pour but de développer une méthode innovante pour prédire les crises d'épilepsie. Grâce à la base de données CHBMIT et à une analyse minutieuse des données, l'équipe de recherche pourra avancer vers une meilleure compréhension de l'épilepsie et proposer une méthode prometteuse pour améliorer la qualité de vie des patients atteints de cette maladie.

## Description de la base de données CHBMIT

La base de données [CHBMIT](https://physionet.org/content/chbmit/1.0.0/) est un ensemble précieux de données médicales qui est accessible pour des recherches dans le domaine de la neurologie et de l'apprentissage automatique. Elle contient des enregistrements EEG de patients pédiatriques atteints d'épilepsie, et est hébergée sur le site PhysioNet. Les données EEG ont été enregistrées auprès de 23 patients, avec plus de 68 heures de données enregistrées pour chaque patient, à travers plusieurs sessions d'une heure chacune. Ces sessions ont été enregistrées à l'aide de 128 canaux EEG.

Il convient de noter que les données sont anonymisées pour protéger la vie privée des patients, ce qui signifie qu'aucune information personnelle identifiable n'est incluse dans les enregistrements. Cela permet aux chercheurs d'accéder à ces données sans violer la confidentialité des patients. Cependant, l'utilisation de ces données est limitée à des fins de recherche médicale.

La base de données CHBMIT est largement utilisée pour la recherche en neurologie et en apprentissage automatique. En particulier, elle est utilisée pour le développement d'algorithmes de détection d'épilepsie et de classification des états de sommeil. Les données EEG contenues dans la base de données sont très utiles pour étudier les différents aspects de l'épilepsie chez les enfants, ce qui peut aider les médecins à mieux comprendre et à mieux traiter cette maladie complexe.

En somme, la base de données CHBMIT est une ressource inestimable pour les chercheurs dans les domaines de la neurologie et de l'apprentissage automatique. Les données anonymisées qu'elle contient permettent aux chercheurs d'approfondir leur compréhension de l'épilepsie chez les enfants et de développer de nouvelles approches de traitement pour cette maladie.

## Analyse des données

### Durée D

Les crises d'épilepsie peuvent survenir en groupes appelés « orages de crises ». Pour résoudre le problème de prédiction, deux types de crises doivent être distinguées : les premières crises et les répliques. La durée minimale sans crise précédant une crise, appelée durée D, est utilisée pour différencier ces deux types. Les données du tableau suivant présentent la proportion de chaque type de crise dans le jeu de données en fonction de la durée D.

| Durée D | Nombre de crises | Pourcentage |
| --- | --- | --- |
| 2h | 49 | 27% |
| 1h30 | 9 | 5% |
| 1h | 22 | 12% |
| 30 min | 27 | 15% |
| 0 min | 75 | 41% |
| Total | 182 | 100% |

Le tableau ci-dessus montre que la majorité des crises (41%) ont lieu dans un intervalle inférieur à 30 minutes. Les durées D de 0 et 2 heures représentent respectivement 41% et 27% des crises, tandis que les autres durées représentent des pourcentages moins élevés, allant de 5% pour une durée de 1h30 à 12% pour une durée de 1h.

### Durée S
Comme mentionné précédemment, la plupart des crises d'épilepsie ont un intervalle de moins de 30 minutes entre elles. Cela soulève la question de savoir s'il faut considérer deux crises séparées ou une seule. Pour répondre à cette question, une durée de séparabilité S est définie. Si la durée entre deux crises est inférieure à S, alors les deux crises sont considérées comme une seule dans le cadre de ce projet.

La table suivant présente les résultats de l'analyse des crises en fonction de la durée S.

| Durée S | Nombre de crises | Pourcentage |
| --- | --- | --- |
| 2 min | 2 | 3% |
| 3 min | 6 | 8% |
| 4 min | 6 | 8% |
| 10 min | 13 | 18% |
| Plus de 10 min | 32 | 43% |
| Total | 74 | 100% |

On constate que plus de la moitié des crises (57%) ont une durée supérieur à 10 minutes. Les durées de séparabilité de 4 à 10 minutes représentent environ un quart des crises, tandis que les durées de 2 à 3 minutes ne représentent qu'une minorité des crises.

## Sélection des données
La sélection des données est une étape cruciale dans tout projet de prédiction. À partir des résultats de l'analyse, il est possible de déterminer les données qui sont pertinentes pour atteindre l'objectif fixé.

Dans le cas présent, l'analyse a mis en évidence que la majorité des crises ont une durée inférieure à 30 minutes et que la plupart des crises ont une durée de séparabilité supérieure à 10 minutes. Ainsi, pour éviter que deux crises soient considérées comme une seule, la durée de séparabilité doit être supérieure à 10 minutes.

En conséquence, pour la suite du projet, nous allons travailler avec les données qui correspondent à une durée D de 2 heures et une durée S de 4 minutes. Ces données permettent d'obtenir un équilibre entre le nombre de crises prises en compte et la durée de séparabilité nécessaire pour éviter la fusion de deux crises en une seule.

Cette sélection de données permettra de construire un modèle de prédiction plus précis et adapté au contexte des patients souffrant d'épilepsie. En outre, cette approche permettra d'optimiser les ressources utilisées en évitant de traiter des données inutiles ou non pertinentes pour le problème de prédiction.

## Conclusion
En conclusion, ce projet a pour objectif de développer un modèle de prédiction de crises d'épilepsie à partir des données EEG de la base de données CHBMIT. L'analyse des données a permis de déterminer deux paramètres importants pour la prédiction des crises : la durée D, qui permet de distinguer les premières crises des répliques, et la durée S, qui permet de déterminer si deux crises successives doivent être considérées comme une seule ou non.

L'analyse des données a également montré que la majorité des crises ont lieu dans un intervalle de moins de 30 minutes et que la durée D de 0 et 2 heures représente la majorité des crises.
