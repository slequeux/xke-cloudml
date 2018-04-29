# Prédictions avec InceptionV3 en local

Vous demandez à Jean comment faire une prédiction sur votre poste pour commencer.
Il vous indique de lancer la commande suivante sur l'une des images du jeu de données dont vous disposez.

```bash
# le paramètre --image correspond à une image de votre système de fichiers local sur laquelle faire une prédiction

python predict/local.py --image /full/path/to/image
```

Vous devriez obtenir un résultat semblable à celui-ci :

```bash
Predicted:
Score 0.523641943932, Label orange
Score 0.23567853868, Label lemon
Score 0.0716699808836, Label Granny_Smith
Score 0.0380598641932, Label banana
Score 0.00295152305625, Label pineapple
```

Allons donc faire un tour du côté du code.
Vous remarquerez la complexité du modèle que Jean utilise :

```python
model = InceptionV3(weights='imagenet')
```

InceptionV3 est un modèle pré-entraîné sur différents jeux de données.
Ici Jean utilise directement InceptionV3 entraîné sur le jeu de données `imagenet`

En fait, Jean ne fait que charger un modèle fourni par défaut par Keras.
Bon, ce n'est pas glorieux, mais nous allons tout de même tenter de packager ce qu'il demande dans le prochain exercice.

Bonus
----

Tentez de lancer la prédiction sur l'image `data/fruits/apple/image_apple1.jpeg`.

Le modèle devrait vous prédire qu'il s'agit d'une orange.
Saurez-vous identifier pourquoi ?