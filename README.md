# FORMATION DLO-JZ : Deep Learning optimisé sur Jean Zay

Depôt pour la formation IA avancée dédiée à Jean Zay :)

## Supports pédagogiques :
http://www.idris.fr/formations/dlo-jz/

## Contenu de la formation :

Cette formation est dédiée au passage à l'échelle multi-GPU de l'entraînement d'un modèle en Deep Learning. Le fil conducteur des aspects pratiques de la formation est la mise à l'échelle optimisée (accélération, distribution) d'un entraînement de modèle sur la base de données Imagenet, en PyTorch. Pour cela, les participant·e·s seront amené.e.s à coder et soumettre leurs calculs sur Jean Zay en appliquant les différentes notions abordées pendant les parties de cours.

## Plan :


### Jour 1

* Accueil
* Présentation de la formation DLO-JZ
* Le supercalculateur Jean Zay
* Les enjeux de la montée à l'échelle
* L'accélération GPU
* La précision mixte
* L'optimisation des formats de tenseur (channels last memory format)
* Profilage de code

### Jour 2

* Optimisation du prétraitement des données sur CPU (DataLoader)
* Entraînement distribué : notions générales et parallélisme de données
* Le stockage et le format des données d'entrée (webdataset)

### Jour 3

* JIT (torch.compile)
* Entraînement large batches (learning rate scheduler, optimiseurs large batches,…)
* Les techniques d'optimisation de recherche des hyperparamètres

### Jour 4

* Gros modèles
* FSDP
* Les parallélismes de modèle
* Les API pour les parallélismes de modèle
* Les outils de visualisation et suivi d'expériences
* Les bonnes pratiques


TP au choix :
* Un exemple de gros modèle : le Vision Transformer
* Tensor Parallelism à la main
* L'augmentation de données (Data Augmentation)
* JIT et compilation

### Quizz
* [TP1_0](https://www.deepmama.com/quizz/dlojz_quizz1.html)
* [TP1_1](https://www.deepmama.com/quizz/dlojz_quizz2.html)
* [TP1_2](https://www.deepmama.com/quizz/dlojz_quizz3.html)
* [TP2_1](https://www.deepmama.com/quizz/dlojz_quizz4.html)
* [TP2_2](https://www.deepmama.com/quizz/dlojz_quizz5.html)

### Videos DP et ZeRODP
* [Animations parallélisme](https://www.youtube.com/playlist?list=PLQd0PPHfzSeJ-gBR8RNfmEEE2ZMBhRb1B)

## Durée :
4 jours.

## Équipement :
Les parties pratiques se dérouleront sur le supercalculateur Jean Zay de l'IDRIS.

## Intervenants :

- Bertrand Cabot
- Nathan Cassereau
- Kamel Guerda
- Léo Hunout

## S’inscrire à cette formation :
https://cours.idris.fr

### Améliorations :
* [ ] FFCV
* [ ] DALI
* [ ] torchdata
* [ ] Transformer, Accelerate, TransformerEngine, PEFT
