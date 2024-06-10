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

### Jour 2

* Entraînement distribué : notions générales et parallélisme de données
* Profilage de code
* Optimisation du prétraitement des données sur CPU (DataLoader)

### Jour 3

* Le stockage et le format des données d'entrée (webdataset)
* Les outils de visualisation et suivi d'expériences
* Entraînement large batches (learning rate scheduler, optimiseurs large batches,…)
* Les techniques d'optimisation de recherche des hyperparamètres

### Jour 4

* JIT (torch.compile)
* Les bonnes pratiques
* Les parallélismes de modèle
* Les API pour les parallélismes de modèle

TP au choix :
* Un exemple de gros modèle : le Vision Transformer
* DDP et TP_DDP à la main
* L'augmentation de données (Data Augmentation)
* JIT et compilation

### Quizz
* [TP1_0](https://www.deepmama.com/quizz/dlojz_quizz1.html)
* [TP1_1](https://www.deepmama.com/quizz/dlojz_quizz2.html)
* [TP1_2](https://www.deepmama.com/quizz/dlojz_quizz3.html)
  

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
* [x] pytorch 2.0
* [ ] DALI
* [ ] torchdata
* [ ] Transformer, Accelerate, TransformerEngine, PEFT
