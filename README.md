#### Projet : Prédiction de la dépression à partir de tweets

Ce projet vise à prédire si une personne est dépressive en analysant un dataset de tweets. Pour ce faire, plusieurs algorithmes d'apprentissage automatique sont utilisés pour entraîner un modèle sur les données et prédire si un tweet donné est émis par une personne dépressive ou non.

#### Prérequis

##### Python 3

Les bibliothèques Python suivantes : pandas, sklearn, matplotlib, numpy
Utilisation

- Téléchargez ou clonez ce dépôt sur votre ordinateur
- Assurez-vous d'avoir installé les prérequis ci-dessus
- Ouvrez un terminal et naviguez jusqu'au répertoire du projet
- Exécutez le script main.py en utilisant la commande python main.py
- Suivez les instructions à l'écran pour choisir un algorithme d'apprentissage et entrer des tweets à analyser

### Fichiers

- data_reader.py : contient une fonction pour lire les données à partir d'un fichier CSV
- preprocessing_data.py : contient des fonctions pour nettoyer et prétraiter les données
- splitting_data.py : contient une fonction pour diviser les données en ensemble d'entraînement et de test
- learning_on_data_random_forest.py : contient une fonction pour entraîner un modèle de forêt aléatoire sur les données
- learning_on_data_svc.py : contient une fonction pour entraîner un modèle SVC (Support Vector Classification) sur les données
- learning_on_data_regression.py : contient une fonction pour entraîner un modèle de régression logistique sur les données
- learning_on_data_anns.py : contient une fonction pour entraîner un réseau de neurones artificiel sur les données
- predicting_on_new_data.py : contient des fonctions pour prédire la dépression sur de nouveaux tweets à l'aide d'un modèle entraîné
- save_data.py : contient une fonction pour enregistrer les données de sortie dans un fichier csv dans un dossier de sortie
- tools.py : contient des fonctions pour évaluer les performances d'un modèle et tracer une matrice de confusion
- main.py : script principal qui orchestre l'ensemble du processus

### Utilisation

Pour exécuter le projet, ouvrez une invite de commande dans le répertoire du projet et exécutez la commande suivante:
` python3 __main__.py`

L'utilisateur sera invité à entrer un numéro pour choisir un méthode d'apprentissage (1: Random Forest, 2: SVC, 3: Logistic Regression, 4: ANNs)

#### Auteurs

###### Ce projet a été réalisé par Manel KHEFFACHE et Ian AVELAR
