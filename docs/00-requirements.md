# Pré-requis

Ce hands'on a été prévu pour être joué lors du XKE d'avril 2018 à Xebia.

```
- export PYTHONPATH=`pwd`
```

## Configuration de GCloud

Il sera nécessaire de disposer des éléments suivants sur Google Cloud
- Un compte Google qui peut accéder à la [console](https://console.cloud.google.com)
  Si vous ne disposez pas d'un tel compte, Google permet la création d'une version d'essai avec un crédit de 300$ disponible pendant 1 an (état de l'offre à date de rédaction de cet execice en avril 2018).
- Sur ce compte, un projet avec la facturation activee
  Dans le cas d'un compte d'essai, cette facturation ira piocher dans les crédits fournis avec le compte.

## Connaissances

- Connaissances de base sur les réseaux de neuronnes
- Connaissances de base sur le Cloud et la notion de service managé

## Outillage

- [Un IDE digne de ce nom](https://www.jetbrains.com/idea/)
- [Anaconda](https://conda.io)
- La [CLI Google Cloud](https://cloud.google.com/sdk/downloads)
- Git


## Préparation

### Récupération du code source

Récupérer le code source de ce projet :

```
git clone git@github.com:slequeux/xke-cloudml.git
```

L'ensemble des commandes seront maintenant exécutées dans le répertoire `xke-cloudml` qui vient d'être créé.

### Création de l'environnement Anaconda

Les exercices suivants nécessitent d'utiliser un environnement Python 2.7 correctement configuré.
Nous utiliserons Anaconda afin de simplifier la gestion des environnements.

```
conda create -n xke-cloudml python=2.7
[...]
Proceed ([y]/n)? y
```

Pour la suite des exercices, l'environnement devra être activé manuellement.
Pour cela, lors de chaque ouverture d'un terminal, lancer la commande suivante :

```
source activate xke-cloudml
```

Avant de commencer les exercices, vous pouvez dors et déjà installer les librairies pré-requises dans l'environnement Anaconda :

```
pip install --upgrade pip
pip install -r requirements.txt
```

### Confirguration de la CLI Google Cloud

La ligne de commande `gcloud` doit définir le projet qui sera utilisé dans ces exercices comme le projet par défaut.
Si vous ne souhaitez pas changer de projet par défaut, il sera nécessaire d'adapter toutes les commandes `gcloud` des différents exercices.

Pour contrôler le projet par défaut :

```
gcloud config get-value project
```

Pour modifier le projet par défaut :

```
gcloud config set project <project id>
```

Attention ici à bien fournir l'ID du projet et non son nom.
L'ID du projet peut être obtenu sur la page d'accueil du projet.

Une fois la CLI configurée, il sera également nécessaire de disposer d'un token d'authentification pour accéder programmatiquement aux APIs.
Pour générer ce token :

```
gcloud auth application-default login
```

Un fichier JSON contenant les tokens sera créé dans votre *home*.

### Récupération du jeu de données d'entraînement

Afin de réaliser certains exercices, nous allons utiliser un jeu de données en local.
Ce jeu de données fait environ 2 Go décompressés.
Pour le récupérer, lancer la commande suivante :

```
python input/download_dataset.py
```



