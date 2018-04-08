# Tensor in the Sky with CloudML

Lundi matin, vous venez de finir votre troisième [kaffe][http://caffe.berkeleyvision.org/] :)
Jean, data scientist, vous interpelle et vous annonce une bonne nouvelle :

"Je viens de mettre en place un nouvel algo qui va améliorer notre projet d'une manière significative.
Par contre, j'ai mon code en local, mais je ne sais pas comment je pourrais faire en sorte de le déployer.
Tu peux m'aider ?"

N'écoutant que votre coeur, vous vous lancez dans cette tâche chevaleresque !

# Pré-requis au Hands'on

Ce hands'on a été prévu pour être joué lors du XKE d'avril 2018 à Xebia.

- Un compte Google Xebia
- [Un IDE digne de ce nom][https://www.jetbrains.com/idea/]
- Un [environnement Anaconda][https://conda.io/docs/user-guide/tasks/manage-environments.html] avec Python 2.7 et tensorflow 1.5.
- La [CLI Google Cloud][https://cloud.google.com/sdk/downloads] installée et authentifiée
  sur le compte Xebia. Choisir `aerobic-coast-161423` comme projet par défaut.
- Cloner le répository Git contenant le README que vous êtes en train de lire :)
- Si vous souhaitez pouvoir réaliser l'entraînement en local, lancer la
  commande `python input/download_dataset.py` qui téléchargera un dataset d'entraînement.
  Cela requiert environ 5 Go d'espace disponible sur disque.
- ```export PYTHONPATH=`pwd` ```

Remarque : l'authentification de la CLI Gcloud avec la commande `gcloud auth application-default login` doit
produire un fichier d'authentification dont vous devez conserver le chemin. Celui-ci sera utile
pendant l'exercice.

# Prédictions avec InceptionV3 en local

Vous demandez à Jean comment faire une prédiction sur votre poste pour commencer.
Il vous indique de lancer la commande `python predict/local.py --image /full/path/to/image` que vous vous empressez de lancer.

Vous devriez obtenir un résultat semblable à celui-ci :
```
Predicted:
Score 0.523641943932, Label orange
Score 0.23567853868, Label lemon
Score 0.0716699808836, Label Granny_Smith
Score 0.0380598641932, Label banana
Score 0.00295152305625, Label pineapple
```

Allons donc faire un tour du côté du code. Vous remarquerez la complexité du modèle que Jean utilise :
```python
model = InceptionV3(weights='imagenet')
```

En fait, Jean ne fait que charger un modèle fourni par défaut par Keras.
Bon, ce n'est pas glorieux, mais nous allons tout de même tenter de packager ce qu'il demande.

# Créer un premier livrable

Une solution très simpliste pour packager les travaux de Jean serait de créer un zip du fichier python.
Mais ici, nous allons tenter d'utiliser un service managé pour déployer son modèle, ceci afin de ne pas avoir à gérer ni infrastructure, ni exposition à travers une connection TCP.

## Exporter Inception V3

Afin d'exporter le modèle Inception V3, vous rédigez un petit script et l'exécutez avec la commande `python export/export_model.py --model_dir /path/to/model/`
Nous allons décrire ce qui est réalisé par ce script :

La fonction `export_current_graph_for_serving` permet de définir une fonction de serving à sauvegarder.
Cette fonction défini deux tenseurs :
- Celui d'entrée : lorsque le service de serving recevra une requête, il injectera les données dans ce tenseur
- Celui de sortie : ce tenseur contient les prédictions.

La fonction `add_base_64_decode_input_layers` permet d'ajouter quelques étapes au graph d'exécution du modèle InceptionV3.
Par défaut, Inception prend en entrée des images sous forme de tableaux de float64.
Voulant réaliser des prédictions qui passeront par internet, nous allons réduire les données transférées en envoyant les images encodées en base 64.
Les étapes ajoutées au graph réalisent cette opération.

## Effectuer une prédiction sur le modèle exporter

Nous allons tester que le modèle répond toujours correctement.
Pour cela, vous complétez le script `predict/local.py` et le lancez maintenant avec la commande `python predict/local.py --image data/fruits/apple/image_apple1.jpeg --model_path models/pure_inception --decode_preds_fn inception`

La prédiction ainsi obtenue devrait être semblable au résultat précédent.

# Prédiction dans le cloud

Il est temps de déplacer le tout dans le cloud. Assez du travail en local, il faut maintenant publier ce modèle.

## Déploiement du modèle

La première étape sera de déposer le fichier `saved_model.pb` dans un répertoire sur GCS.

Une fois cela fait, redez-vous sur le service CloudML.
Il suffit maintenant de créer un modèle en lui donnant un nom et une description.
Dans ce modèle, créez une nouvelle version en pointant vers le répertoire contenant le fichier protobuff, et en spécifiant la version de runtime 1.5.

Après quelques secondes (voir minutes) d'attente, la première version du modèle sera déployée.

## Test du modèle

Il est maintenant temps d'écrire un script pour tester le modèle qui vient d'être déployé.
Ce script sera `python predict/gcloud.py --image data/fruits/apple/image_apple1.jpeg --project_id aerobic-coast-161423 --model_name simple_mnist --model_version v2 --decode_preds_fn inception --top 6`
La sortie devrait à nouveau avoir un résultat semblable.

# Un nouveau modèle

Paul a entendu parler de vos exploit avec cloud-ml.
Il souhaiterais bénéficier de la puissance du cloud pour entrainer son modèle.
Pour lancer son modèle Paul vous a donner la commande ci-dessous. Il vous a aussi précisé qu'il utilisait tensorflow 1.4

```
python task.py \
    --job-dir $MODEL_DIR \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100

```

## Entraînement dans le Cloud

Les données d'entrainement et d'évaluation sont déja sur google storage.
On souhaite que le model entrainé soit stocké sur google storage dans le dossier paul/model/my_model.
On utilise la region europe-west1.
```
TRAIN_DATA=gs://sleq-ml-engine/paul/data/adult.data.csv
EVAL_DATA=gs://sleq-ml-engine/paul/data/adult.test.csv
JOB_NAME=my_model
OUTPUT_PATH=gs://sleq-ml-engine/paul/model/$JOB_NAME
REGION=europe-west1
```

pour savoir comment lancer un entrainement dans le cloud aider vous de la commande suivante.
```
gcloud ml-engine jobs submit training --help
```

lors d'un entrainement dans le cloud vous mettez d'abord les paramètre pour configurer cloud-ml puis vous utilisez le séparateur `--` et vous metez les paramètre à passer à votre programme python.
cependant le --job-dir et un paramètre un peu spécial il faut le passer dans le paramètre cloud-ml mais il sera aussi automatiquement passer à votre programme.


## Création d'une nouvelle version du modèle

L'entraînement du modèle dans le cloud à générer un fichier saved_model.pb dans GCS.
Vous pouvez donc maintenant créer une nouvelle version du modèle dans CloudML.

Afin de la tester, vous pouvez utiliser le script `python predict/gcloud.py --image data/fruits/apple/image_apple1.jpeg --project_id aerobic-coast-161423 --model_name simple_mnist --model_version v4 --decode_preds_fn transfert --top 6`

## Bascule de modèle sans interruption de service

La nouvelle version du modèle est jugée satisfaisante pour Jean.
Il ne vous reste plus qu'à cliquer sur le bouton `Définir en tant que version par défaut` dans la liste des versions pour changer la version utilisée par défaut (celle proposée aux utilisateurs).

# (Optionnel et non guidé) Vers l'infini et l'au-delà

Jean a maintenant la possibilité de facilement déployer des nouvelles versions de son modèle.
Mais jouer avec ce service vous a donner de nombreuses idées que vous explorerez dans le futur.

## CI/CD de modèles de deep learning

Jean travaille sur des fichiers Python. Il s'agit donc de code source, et probablement, ce code est versionné.
Vous allez donc pouvoir vous assurer que tout commit donnera lieu à une version déployabe et testable.

Il devient plus facile de faire de la CI/CD sur du deep learning.

## Prédictions en HTTP

Pour l'instant, les appels à l'API de prédiction sont réalisés en Python.
Vous aimeriez pouvoir envoyer les images via des requêtes HTTP pour récupérer les prédictions.

Peut-être qu'une Google Cloud function bien paramétrée répondrait au problème.

A vous de jouer !