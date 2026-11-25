# HSMT et reconnaissance de motifs textiles

Le manuscrit révisé est fourni en PDF et sous forme de sources LaTeX. Cette extension porte sur la décomposition quadratique de HSMT. Le supplément antérieur de l’article conserve les expériences DTD, KTH et fleurs.

## Résultat scientifique

Sur les 211 photographies retenues, 30 partitions stratifiées en cinq plis donnent un rappel macro moyen de 61,68 % pour HSMT avec carrés, de 53,62 % avec produits croisés et de 52,23 % pour la somme pondérée. L’écart apparié entre carrés et produits croisés atteint 8,05 points en moyenne. Il est positif sur les 30 partitions. Son intervalle à 95 % par rééchantillonnage des photographies au sein des classes va de 4,90 à 11,37 points. Les 30 décisions d’une même photographie restent ensemble. L’intervalle est conditionnel aux modèles ajustés. Ces partitions partagent les mêmes 211 sources.

Le CLBP conjoint en niveaux de gris atteint 60,19 % et le DINOv2 ViT-S/14 gelé atteint 88,15 %. Avec un poids fixe de moitié pour chaque noyau normalisé, DINOv2 et HSMT avec carrés atteignent ensemble 86,85 %. Les valeurs ont été arrondies après calcul de chaque contraste. Le dossier contient les deux hybrides, les six comparaisons prévues et les résultats des sept modèles.

La collection compte 14 photographies unies, 23 florales, 34 en zigzag, 39 rayées, 43 à carreaux et 58 à pois. Les classes reçoivent le même poids quel que soit leur effectif. La sélection initiale portait sur 720 entrées du manifeste. La disponibilité des sources, les contrôles de géométrie, les annotations contradictoires et les doublons ont déterminé la cohorte finale. L’unité de validation est la photographie source. L’identité d’un même vêtement photographié plusieurs fois n’est pas renseignée.

## Lire le dossier

- `PROTOCOL.txt` et `LOCK.sha256` donnent le protocole et les empreintes enregistrés avant de calculer les scores sur les 30 nouvelles partitions. Le premier examen textile avait déjà été effectué.
- `results/paired.csv` donne les écarts en points, leurs intervalles nominaux à 95 %, les intervalles ajustés pour la famille des cinq comparaisons secondaires, ainsi que le nombre de partitions dans chaque sens.
- `results/by_partition.csv` et `results/by_class.csv` donnent les rappels en pourcentage. `results/predictions.csv` contient une ligne par photographie et partition. `results/predictions.npz` conserve les grilles de validation et les indices choisis.
- `data/sources.csv` identifie les pixels de chaque recadrage RGB. `data/partitions.npz` archive toutes les affectations. `data/features.npz` et `data/reference_features.npz` permettent de refaire les ajustements hors ligne.
- `results/pixel_audit.json` compare tous les descripteurs HSMT archivés à une extraction fraîche sur les 211 images. Toutes les coordonnées F7, F8 et RGB concordent à la précision stockée.
- Le complément scientifique se trouve dans `../reader/HSMT_PR_Supplement.pdf`. Le tableau et la figure textiles utilisent `code/paper_table_figure.py`.

## Refaire le calcul depuis les descripteurs

Après extraction, exécuter ces commandes depuis le sous-dossier `clothing/`. La version utilisée était Python 3.12. Les dépendances de calcul sont NumPy 2.3.5, SciPy 1.17.0 et threadpoolctl 3.6.0. La reconstruction des partitions utilise scikit-learn 1.8.0.

```bash
python code/verify_previous.py
python code/evaluate_partitions.py
python code/summarize.py
python code/check_results.py
```

La première commande vérifie les 633 décisions des trois variantes HSMT sur la partition historique reconstruite. La dernière recalcule les rappels à partir du CSV, vérifie la sélection des paramètres et contrôle l’identité algébrique du manuscrit. Le script de figure requiert également Matplotlib. La source éditable du manuscrit est fournie séparément dans HSMT_PR_LaTeX.zip.

## Repartir des photographies

Le dossier conserve les descripteurs, les URL, les fenêtres de recadrage et les empreintes. Les photographies et les poids du modèle restent chez leurs fournisseurs. La récupération peut être partielle si une URL a changé. Les calculs depuis les descripteurs restent possibles dans ce cas. L’extraction complète emploie Pillow 13.0.0, PyTorch CPU 2.5.1, pytorch-wavelets 1.3.0 et un compilateur C++17 avec OpenMP.

```bash
python code/recover_crops.py photos
python code/check_pixels.py photos
python code/extract_refs.py photos weights
```

Le répertoire `weights` doit contenir `dino.pth` (modèle officiel sans registres) et `dinov2.zip` (archive du code épinglé). Les deux empreintes sont vérifiées avant chargement. La réextraction des références peut être omise pour l’analyse des scores.

- [Manifeste des motifs textiles](https://github.com/lstearns86/clothing-pattern-dataset)
- [Guo, Zhang et Zhang, CLBP](https://doi.org/10.1109/TIP.2010.2044957)
- [Oquab et al., DINOv2](https://arxiv.org/abs/2304.07193)
- [Poids officiels ViT-S/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)
- [Archive du code DINOv2](https://codeload.github.com/facebookresearch/dinov2/zip/e1277af2ba9496fbadf7aec6eba56e8d882d1e35)
