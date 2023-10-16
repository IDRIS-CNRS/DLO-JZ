# FORMATION DLO-JZ : Deep Learning optimisé sur Jean Zay

Depôt pour la formation IA avancée dédiée à Jean Zay :)

## Supports pédagogiques :
http://www.idris.fr/formations/dlo-jz/

## Contenu de la formation :

Cette formation est dédiée au passage à l'échelle multi-GPU de l'entraînement d'un modèle en Deep Learning. Le fil conducteur des aspects pratiques de la formation est la mise à l'échelle optimisée (accélération, distribution) d'un entraînement de modèle sur la base de données Imagenet, en PyTorch. Pour cela, les participant·e·s seront amené.e.s à coder et soumettre leurs calculs sur Jean Zay en appliquant les différentes notions abordées pendant les parties de cours.

## Plan :

:checkered_flag: : Imagenet Race CPU

[voir résultats sur Weight and Biases](https://wandb.ai/dlojz/Imagenet%20Race%20Cup?workspace=user-bcabot)

### Jour 1

* Accueil
* Présentation de la formation DLO-JZ
* Le supercalculateur Jean Zay
* Les enjeux de la montée à l'échelle
* L'accélération GPU
* La précision mixte
* L'optimisation des formats de tenseur (channels last memory format)

### Jour 2

* Entraînement distribué : notions générales et parallélisme de données
* :checkered_flag: Lancement d'un premier entraînement complet sur 32 GPU V100 (qui tournera pendant la nuit)
* Profilage de code
* Optimisation du prétraitement des données sur CPU (DataLoader)
* Entraînement large batches (learning rate scheduler, optimiseurs large batches,…)

### Jour 3

* :checkered_flag: Visualisation des résultats du premier entraînement lancé la veille
* :checkered_flag: Optimisation et lancement d'un second entraînement sur 32 GPU V100 (qui tournera pendant la nuit)
* Les techniques d'optimisation de recherche des hyperparamètres
* Le stockage et le format des données d'entrée (webdataset)
* L'augmentation de données (Data Augmentation)

### Jour 4

* :checkered_flag: Visualisation des résultats du second entraînement lancé la veille
* Les bonnes pratiques
* Les parallélismes de modèle
* Les API pour les parallélismes de modèle
* Un exemple de gros modèle : le Vision Transformer


## Durée :
4 jours.

## Équipement :
Les parties pratiques se dérouleront sur le supercalculateur Jean Zay de l'IDRIS.

## Intervenants :

- Bertrand Cabot
- Nathan Cassereau
- Kamel Guerda
- Léo Hunout
- Myriam Peyrounette

## S’inscrire à cette formation :
https://cours.idris.fr

### Améliorations :
- pytorch 2.0
- DALI
- torchdata
- Transformer, Accelerate, TransformerEngine, PEFT
