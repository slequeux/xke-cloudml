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

Remarque : l'authentification de la CLI Gcloud avec la commande `gcloud auth application-default login` doit
produire un fichier d'authentification dont vous devez conserver le chemin. Celui-ci sera utile
pendant l'exercice.

# Prédictions avec InceptionV3 en local

Vous demandez à Jean comment faire une prédiction sur votre poste pour commencer.
Il vous indique de lancer la commande `python predict/local_inception.py --image /full/path/to/image` que vous vous empressez de lancer.

Vous devriez obtenir un résultat semblable à celui-ci :
```
Predicted:
Score 0.412874519825, Label candle
Score 0.0934453085065, Label lipstick
Score 0.0552788861096, Label face_powder
Score 0.0318395756185, Label hair_spray
Score 0.0220600552857, Label bottlecap
```

Allons donc faire un tour du côté du code. Vous remarquerez la complexité du modèle que Jean utilise :
```python
def load_inception():
    return InceptionV3(weights='imagenet')
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


TODO à rédiger

# Prédiction dans le cloud

TODO à rédiger

# Un nouveau modèle

TODO à rédiger => précision sur le temps d'entraînement sur un MBP (env 3h)
```
Loading base model ...
Loaded
Loading input dataset
Loading category fruits
Found 9 labels ['apple', 'strawberry', 'kiwi', 'raspberry', 'mango', 'banana', 'grape', 'pineapple', 'orange']
Loading label apple
	Found 1033 examples
	Loaded
Loading label strawberry
	Found 1172 examples
	Loaded
Loading label kiwi
	Found 1149 examples
	Loaded
Loading label raspberry
	Found 1243 examples
	Loaded
Loading label mango
	Found 1016 examples
	Loaded
Loading label banana
	Found 1111 examples
	Loaded
Loading label grape
	Found 1354 examples
	Loaded
Loading label pineapple
	Found 936 examples
	Loaded
Loading label orange
	Found 1177 examples
	Loaded
Category fruits loaded
Defining the model
Training ...
Train on 7643 samples, validate on 2548 samples
Epoch 1/3
 - 3253s - loss: 0.9586 - acc: 0.6915 - val_loss: 12.3438 - val_acc: 0.1252
Epoch 2/3
 - 2717s - loss: 0.6516 - acc: 0.7909 - val_loss: 13.4497 - val_acc: 0.1240
Epoch 3/3
 - 3181s - loss: 0.6022 - acc: 0.7979 - val_loss: 13.2632 - val_acc: 0.1287

  32/2548 [..............................] - ETA: 13:38
  64/2548 [..............................] - ETA: 12:45
  96/2548 [>.............................] - ETA: 12:07
 128/2548 [>.............................] - ETA: 11:41
 160/2548 [>.............................] - ETA: 12:11
 192/2548 [=>............................] - ETA: 12:24
 224/2548 [=>............................] - ETA: 12:01
 256/2548 [==>...........................] - ETA: 11:45
 288/2548 [==>...........................] - ETA: 11:48
 320/2548 [==>...........................] - ETA: 11:28
 352/2548 [===>..........................] - ETA: 11:27
 384/2548 [===>..........................] - ETA: 11:16
 416/2548 [===>..........................] - ETA: 11:03
 448/2548 [====>.........................] - ETA: 10:51
 480/2548 [====>.........................] - ETA: 10:39
 512/2548 [=====>........................] - ETA: 10:24
 544/2548 [=====>........................] - ETA: 10:09
 576/2548 [=====>........................] - ETA: 9:57
 608/2548 [======>.......................] - ETA: 9:44
 640/2548 [======>.......................] - ETA: 9:31
 672/2548 [======>.......................] - ETA: 9:19
 704/2548 [=======>......................] - ETA: 9:08
 736/2548 [=======>......................] - ETA: 8:57
 768/2548 [========>.....................] - ETA: 8:45
 800/2548 [========>.....................] - ETA: 8:35
 832/2548 [========>.....................] - ETA: 8:23
 864/2548 [=========>....................] - ETA: 8:13
 896/2548 [=========>....................] - ETA: 8:02
 928/2548 [=========>....................] - ETA: 7:51
 960/2548 [==========>...................] - ETA: 7:41
 992/2548 [==========>...................] - ETA: 7:31
1024/2548 [===========>..................] - ETA: 7:21
1056/2548 [===========>..................] - ETA: 7:11
1088/2548 [===========>..................] - ETA: 7:01
1120/2548 [============>.................] - ETA: 6:51
1152/2548 [============>.................] - ETA: 6:41
1184/2548 [============>.................] - ETA: 6:32
1216/2548 [=============>................] - ETA: 6:22
1248/2548 [=============>................] - ETA: 6:12
1280/2548 [==============>...............] - ETA: 6:03
1312/2548 [==============>...............] - ETA: 5:53
1344/2548 [==============>...............] - ETA: 5:44
1376/2548 [===============>..............] - ETA: 5:34
1408/2548 [===============>..............] - ETA: 5:25
1440/2548 [===============>..............] - ETA: 5:16
1472/2548 [================>.............] - ETA: 5:06
1504/2548 [================>.............] - ETA: 4:57
1536/2548 [=================>............] - ETA: 4:48
1568/2548 [=================>............] - ETA: 4:38
1600/2548 [=================>............] - ETA: 4:29
1632/2548 [==================>...........] - ETA: 4:20
1664/2548 [==================>...........] - ETA: 4:11
1696/2548 [==================>...........] - ETA: 4:02
1728/2548 [===================>..........] - ETA: 3:52
1760/2548 [===================>..........] - ETA: 3:43
1792/2548 [====================>.........] - ETA: 3:34
1824/2548 [====================>.........] - ETA: 3:25
1856/2548 [====================>.........] - ETA: 3:16
1888/2548 [=====================>........] - ETA: 3:07
1920/2548 [=====================>........] - ETA: 2:57
1952/2548 [=====================>........] - ETA: 2:48
1984/2548 [======================>.......] - ETA: 2:39
2016/2548 [======================>.......] - ETA: 2:30
2048/2548 [=======================>......] - ETA: 2:21
2080/2548 [=======================>......] - ETA: 2:12
2112/2548 [=======================>......] - ETA: 2:03
2144/2548 [========================>.....] - ETA: 1:53
2176/2548 [========================>.....] - ETA: 1:44
2208/2548 [========================>.....] - ETA: 1:35
2240/2548 [=========================>....] - ETA: 1:26
2272/2548 [=========================>....] - ETA: 1:17
2304/2548 [==========================>...] - ETA: 1:08
2336/2548 [==========================>...] - ETA: 59s
2368/2548 [==========================>...] - ETA: 50s
2400/2548 [===========================>..] - ETA: 41s
2432/2548 [===========================>..] - ETA: 32s
2464/2548 [============================>.] - ETA: 23s
2496/2548 [============================>.] - ETA: 14s
2528/2548 [============================>.] - ETA: 5s
2548/2548 [==============================] - 714s 280ms/step
Loss 13.2631947085, Accuracy 0.128728414489
Exporting ...
Converted 380 variables to const ops.
End : took 3:00:45.203420
```

## Entraînement dans le Cloud

TODO à rédiger

## Création d'une nouvelle version du modèle

TODO à rédiger

## Bascule de modèle sans interruption de service

TODO à rédiger

# (Optionnel et non guidé) Vers l'infini et l'au-delà

## CI/CD de modèles de deep learning

TODO à rédiger

Ce que l'on pourrait faire en plus :
- Sur un trigger de commit GIT
  - Entraineement du modèle
  - Déploiement de la nouvelle version
  - Tests fonctionnels automatisés
  - Bascule du nouveau modèle "par défaut"
Et BIM, CI/CD sur du Deep Learning !

## Prédictions en HTTP

(Optional) Création d'une lambda pour répondre à un POST HTTP