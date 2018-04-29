# Créer un premier livrable

Une solution très simpliste pour packager les travaux de Jean serait de créer un zip du fichier python.
Mais ici, nous allons tenter d'utiliser un service managé pour déployer son modèle, ceci afin de n'avoir à gérer ni infrastructure, ni exposition à travers une connection TCP.

## Exporter Inception V3

Le service managé que nous allons utiliser est CloudML.
Ce service permet d'exposer des modèles déjà entraînés. On appelle cette phase le *serving**.
CloudML effectuera donc le serving, mais uniquement pour des modèles entraînés avec TensorFlow.

La première étape est de sauvegarder le modèle dans un format supporté par CloudML.
Pour cela vous rédigez un petit script et l'exécutez avec la commande suivante :
```
# /path/to/model est le répertoire de destination du modèle sauvegardé.

python export/export_model.py --model_dir /path/to/model/
```

Nous allons maintenant décrire ce qui est réalisé par ce script.

La fonction `export_current_graph_for_serving` permet de définir une fonction de serving à sauvegarder.
Cette fonction défini deux tenseurs :
- Celui d'entrée : lorsque le service de serving recevra une requête, il injectera les données dans ce tenseur
- Celui de sortie : ce tenseur contient les prédictions.

Par défaut, Inception prend en entrée des images sous forme de tableaux de float64.
Transférer une image par ce moyen serait coûteux et nous allons donc les faire transiter dans un autre format.
La fonction `add_base_64_decode_input_layers` permet d'ajouter quelques étapes au graph d'exécution du modèle InceptionV3 afin de transformer un input encodé en base 64 dans le format cible d'Inception.

Après l'exécution de ce script, le répertoire de destination du modèle contient plusieurs éléments :
- Un fichier `saved_model.pb` contenant toutes les constantes du modèle : le graph et les poids entraînés
- Un répertoire `variables` qui contient les variables.
  Notre modèle n'en contient pas et ce répertoire sera donc vide.

## Effectuer une prédiction sur le modèle exporter

Nous allons tester que le modèle répond toujours correctement.
Pour cela, vous lancez la commande suivante :
```
# le paramètre --image correspond à une image de votre système de fichiers local sur laquelle faire une prédiction
# le paramètre --model_path correspond au répertoire dans lequel vous avez exporté le modèle
# le paramètre --decode_preds_fn permet de formater la sortie sur la console

python predict/local.py --image /path/to/image --model_path /path/to/model --decode_preds_fn inception
```

La prédiction ainsi obtenue devrait être semblable au résultat précédent.

# Prédiction dans le cloud

Il est temps de déplacer le tout dans le cloud.
Assez du travail en local, il faut maintenant publier ce modèle.

## Déploiement du modèle

La première étape sera de déposer le fichier `saved_model.pb` dans un répertoire sur GCS.

Pour cela :
- Dans le menu déroulant en haut à gauche, cliquer sur `stockage/navigateur`
- Créer un bucket ou bien sélectionnez un bucket déjà existant.
  Les paramètres de création ne sont pas impactants dans un premier temps.
  Ils le deviendront quan vous souhaiterez optimiser les coûts et la réplication liés à ce modèle.
- Créer un répertoire `models/jean_inception/v1`
- Importez le fichier `saved_model.pb` dans ce répertoire

Une fois cela fait, vous pouvez déclarer votre modèle :
- Dans le menu déroulant en haut à gauche, cliquer sur `ML Engine/modèles`
- Créer un modèle que vous nommerez `jean_inception`.
  Un modèle correspond à un cas d'usage précis.
  Ici, le modèle de Jean consiste à prédire le contenu d'une image.
- Dans ce modèle, créez une version `v1` qui pointe vers le répertoire `<bucket>/models/jean_inception/v1` et utilisant le runtime 1.5.
  Une version de modèle correspond à un modèle différent (au sens du modèle de machine learning).

Après quelques secondes (voir minutes) d'attente, la première version du modèle sera déployée.

## Test du modèle

Il est maintenant temps d'écrire un script pour tester le modèle qui vient d'être déployé.

Appelez ce script à l'aide de la commande suivante :
```
# le paramètre --image correspond à une image de votre système de fichiers local sur laquelle faire une prédiction
# le paramètre --project_id correspond à l'ID du projet GCloud
# le paramètre --model_name correspond au nom du modèle déclaré dans CloudML
# le paramètre --model_version correspond à la version du modèle à utiliser
# le paramètre --decode_preds_fn permet de formater la sortie sur la console
# le paramètre --top défini le nombre d'élements à afficher dans la prédiction

python predict/gcloud.py --image /path/to/image --project_id projectid --model_name jean_inception --model_version v1 --decode_preds_fn inception --top 6
```

La sortie devrait à nouveau avoir un résultat semblable.
