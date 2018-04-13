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