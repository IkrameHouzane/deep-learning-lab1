# Lab 1 – Deep Learning : régression et classification avec PyTorch

Université Abdelmalek Essaadi – FST Tanger – Module Deep Learning et Applications (sous la supervision de: Pr. Elaachak Lotfi)

Auteure : Ikrame Houzane

## Objectif

Me familiariser avec PyTorch en construisant des MLP pour deux tâches :
1. **Régression** : prédire le prix de l'action AAPL (dataset NYSE).
2. **Classification multi-classes** : prédire le type de panne de machines (maintenance prédictive).

Pour chaque partie : EDA, prétraitement, MLP, GridSearch, courbes Loss/Epochs, régularisation, comparaison avec le premier modèle.

## Contenu du dépôt

- `lab1_deep_learning.ipynb` : notebook complet (code, sorties, commentaires et synthèse).

## Données

- [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse) (fichier `prices-split-adjusted.csv`)
- [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

Les datasets ne sont pas dans le dépôt.

## Exécution

Le notebook a été exécuté sur **Kaggle** : ajouter les deux datasets via *Add Input*, puis *Run All*. Les chemins commencent par `/kaggle/input/...` et sont à adapter hors Kaggle.
Bibliothèques : pandas, numpy, matplotlib, seaborn, scikit-learn, PyTorch, imbalanced-learn (`pip install imbalanced-learn` si besoin).

---

## Partie 1 : régression (AAPL)

**EDA.** 851 264 lignes, 501 entreprises, 2010-2016, aucune valeur manquante ni doublon. 21 lignes à `volume = 0` supprimées. Les prix open/high/low/close sont quasi parfaitement corrélés. Les distributions de `close` et `volume` sont très asymétriques (le log les rend plus symétriques). J'ai restreint l'étude à **AAPL** pour que le GridSearch reste faisable.

**Tentative 1 : prédire le prix de demain.** Cible = `close(t+1)`, découpage par date (train 2010-2015, test 2016) pour éviter la fuite de données. Référence : modèle naïf `close(t+1) = close(t)` (MAE 1,05 $). Le MLP obtient 1,09 $, donc ne fait pas mieux. Deux problèmes identifiés : la série est non stationnaire (les entrées de 2015 sont à +2,49 écarts-types de celles de l'entraînement) et close(t) est presque égal à close(t+1) (corrélation 0,9989), ce qui rend le score trompeur.

**Tentative 2 : prédire le rendement du lendemain.**
- Cible : `close(t+1) / close(t) − 1`. Entrées : rendement du jour, variation close/open, amplitude (high − low)/close, variation du log du volume.
- Découpage par date : train 2010-2014, validation 2015, test 2016. Standardisation apprise sur le train uniquement.
- MLP 4 → 64 → 32 → 1 (2 433 paramètres), `MSELoss`, Adam.
- Le modèle de base (100 epochs) surapprend : la loss de validation passe de 1,00 à 1,07, au-dessus du modèle naïf (1,0008).
- **GridSearch** : 16 combinaisons (architecture, learning rate, epochs, optimiseur) avec `ParameterGrid`, évaluées sur la validation 2015. Meilleur réglage : (64, 32), lr = 1e-3, Adam, 10 epochs (loss 0,9829).
- La courbe « Accuracy / Epochs » n'a pas de sens en régression : je l'ai remplacée par le MAE en dollars par epoch.

| Modèle (validation 2015) | Loss |
|---|---|
| Naïf (rendement = 0) | 1,0008 |
| MLP de base, 100 epochs | ≈ 1,07 |
| + Early stopping | 0,9829 |
| + Dropout (0,3) + early stopping | 0,9839 |
| + L2 (1e-2) + early stopping | 0,984 |

**Test 2016** (early stopping seul) : MAE = 1,041 $ contre 1,047 $ pour le modèle naïf. L'écart est trop faible pour conclure à une vraie supériorité : le rendement quotidien est dominé par le bruit.

---

## Partie 2 : classification multi-classes

**Données et nettoyage.** 10 000 lignes. 27 lignes avaient des étiquettes contradictoires entre `Target` et `Failure Type` : supprimées. La classe « Random Failures » disparaît, il reste **5 classes** et 9 973 lignes, dont 96,7 % de « No Failure ». `UDI`, `Product ID` et `Target` (fuite d'information) ne sont pas utilisées comme entrées.

**EDA.** Les boxplots et le nuage couple/rpm montrent qu'aucune mesure ne sépare toutes les classes. Power Failure se trouve aux deux extrémités de la courbe couple/rpm. J'ai donc ajouté trois variables : `power` (couple × vitesse), `temp_diff` (écart de température) et `wear_torque` (usure × couple). `Type` est encodé en one-hot : **11 entrées**.

**Méthode.**
- Découpage stratifié : test 20 %, puis validation 20 % du train ; standardisation apprise sur le train.
- **SMOTE sur le train uniquement**, après le découpage (6 171 exemples par classe). La validation et le test gardent la répartition réelle.
- MLP 11 → 64 → 32 → 5 (3 013 paramètres), `CrossEntropyLoss`, Adam.
- **GridSearch** : 16 combinaisons, critère **F1 macro** sur la validation (l'accuracy est trompeuse : 96,7 % pour un modèle qui répond toujours « No Failure »). Meilleur réglage : (64, 32), lr = 1e-3, Adam, 50 epochs (F1 macro 0,704). Adam est bien meilleur que SGD (≤ 0,40).

| Modèle (validation, F1 macro) | Score |
|---|---|
| Base, 50 epochs | 0,7044 |
| + Early stopping (patience 15) | 0,7258 |
| + Dropout (0,3) | 0,6855 |
| + L2 (1e-4 / 1e-3 / 1e-2) | 0,7224 / 0,6490 / 0,4698 |

Modèle retenu : early stopping seul (le plus simple, les autres écarts sont dans le bruit).

**Test (1 995 lignes).** Accuracy 97,3 % (modèle trivial : 96,7 %), F1 macro 0,687.

| Classe | Précision | Sensibilité | F1 | Exemples |
|---|---|---|---|---|
| No Failure | 0,994 | 0,979 | 0,987 | 1 929 |
| Heat Dissipation | 0,559 | 0,864 | 0,679 | 22 |
| Power | 0,783 | 0,947 | 0,857 | 19 |
| Overstrain | 0,667 | 0,875 | 0,757 | 16 |
| Tool Wear | 0,118 | 0,222 | 0,154 | 9 |

Sur le train (lignes réelles), le F1 macro est de 0,883 : le modèle surapprend. Tool Wear est la classe ratée : elle se confond avec les machines saines et plus de 99 % de ses exemples d'entraînement sont synthétiques.

---

## Ce que j'ai appris

- **PyTorch** : construire un MLP (`nn.Module`), la boucle d'entraînement (forward → loss → backward → step), les modes `train` / `eval`, `MSELoss` en régression, `CrossEntropyLoss` (logits, sans softmax dans le réseau) en classification.
- **Séries temporelles** : découpage par date pour éviter la fuite de données, comparaison systématique à un modèle naïf, et choix d'une cible stationnaire (le rendement plutôt que le prix).
- **Surapprentissage** : lire les courbes train/validation, early stopping. Le dropout et le L2 ne sont pas automatiquement utiles : un L2 trop fort a fait chuter le F1 macro de 0,73 à 0,47.
- **Choix des hyperparamètres** : sur la validation, jamais sur le test (utilisé une seule fois à la fin).
- **Classes déséquilibrées** : l'accuracy est trompeuse ; précision, sensibilité, F1 par classe ; SMOTE après le découpage, uniquement sur le train.
