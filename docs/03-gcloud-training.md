# Un nouveau modèle
Impressionner par vos exploits avec cloud-ml.
Jean souhaiterait bénéficier de la puissance du cloud pour entrainer son modèle.

## Entraînement dans le Cloud
Pour l'entrainement les données doivent être envoyées dans un bucket google cloud storage.
```
gsutil -m rsync -r -J data/fruits gs://$BUCKET/$DATA_PATH
```

On souhaite que le modèle entrainé soit stocké sur google storage dans le dossier model/my_model.
```
VERSION=v2
PROJECT_ID=aerobic-coast-161423
BUCKET=sleq-ml-engine
DATA_PATH=data/raw/fruits/
MODEL_NAME=magritte
JOB_NAME=${MODEL_NAME}_$VERSION
OUTPUT_PATH=gs://sleq-ml-engine/models/$JOB_NAME
```

Le code pour l'entrainement dans le cloud se trouve dans le dossier `cloud_train`.
Pour lancer un entrainement, aidez-vous de la commande suivante.
```
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.5 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region europe-west1 \
    --scale-tier basic-gpu \
    -- \
    --num-epochs 3 \
    --steps-per-epoch 3 \
    --data-project $PROJECT_ID \
    --data-bucket $BUCKET \
    --data-path $DATA_PATH
```
Lors d'un entrainement dans le cloud vous mettez d'abord les paramètres pour configurer cloud-ml puis vous utilisez le séparateur `--` et vous mettez les paramètres à passer à votre programme python.
Le paramètre `--job-dir` est un peu spécial il faut le passer dans les paramètres cloud-ml mais il sera aussi automatiquement passé à votre programme.

Vous pouvez suivre l'état de votre job dans la console dans ML Engine -> Tâches. Il faut environ 5 minutes pour que l'environnement soit installé et que l'apprentissage commence.

## Création d'une nouvelle version du modèle
L'entraînement du modèle dans le cloud a généré un fichier saved_model.pb dans GCS, dans le dossier ${OUTPUT_PATH}/modèle.
Vous pouvez donc maintenant créer une nouvelle version du modèle dans CloudML.
Afin de la tester, vous pouvez utiliser le script 
``` 
predict/gcloud.py \
    --image data/fruits/apple/image_apple1.jpeg \
    --project_id $PROJECT_ID \
    --model_name $MODEL_NAME \
    --model_version $VERSION \
    --decode_preds_fn transfert \
    --top 6
```

## Bascule de modèle sans interruption de service
La nouvelle version du modèle est jugée satisfaisante pour Jean.
Il ne vous reste plus qu'à cliquer sur le bouton `Définir en tant que version par défaut` dans la liste des versions pour changer la version utilisée par défaut (celle proposée aux utilisateurs).