# Dépendances couleur–ondelettes pour la reconnaissance de textures

Cette archive accompagne l’essai de deux mesures de 30 valeurs à comparer au bloc RGB 64 du manuscrit. Les expériences utilisent le profil à trois similarités de même poids et la régression ridge à noyau quadratique complet, λ = 1. Les gains sont en points de macro-rappel.

## Reproduire les décisions et les tableaux hors ligne

Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 et threadpoolctl 3.6.0 ont été utilisés. À partir du dossier extrait, sur un poste avec environ 3 à 5 Go de mémoire disponible pour un seul processus de calcul, exécuter

```bash
python -m pip install -r requirements-replay.txt
python replay.py --threads 2 --jobs 1 --log replay_local.json
```

Le programme réajuste les modèles à partir des descripteurs inclus, régénère les résultats à l’intérieur d’un répertoire temporaire et les compare aux sorties de référence. Les paramètres de chaque pli utilisent ses images d’entraînement. Les étiquettes de référence ne sont utilisées que pour les comparaisons finales ou comme source des prédictions du comparateur déjà réajusté dans le contrôle post-hoc. Les commandes échouent sur une divergence d’étiquettes. Les valeurs numériques des tableaux sont comparées à 10⁻⁹ près. Un système disposant de plus de mémoire peut traiter plusieurs protocoles avec `--jobs 2`. Pour contrôler l’archive elle-même, utiliser `sha256sum -c SHA256SUMS`.

## Plans, comparaisons et résultats

- `PLAN.md`, `PLAN_ALOT.md` et leurs fichiers d’horodatage conservent les deux plans successifs. Le suivi ALOT a été fixé après les résultats KTH. `POIDS_RGB.md` décrit le témoin d’atténuation de RGB ajouté après les résultats ALOT. Les valeurs de DTD, KTH et Lytro du modèle RGB étaient déjà connues avant cette étude.
- `results/scores.csv` fournit le macro-rappel, l’exactitude top-1, la macro-précision et les dimensions. `by_class.csv` et `by_fold.csv` détaillent les classes et les plis. Le nombre de coordonnées est celui des blocs d’entrée au noyau. Une constante supplémentaire dans chaque carte numérique centrée permet un cosinus défini pour une entrée nulle.
- `primary.csv` compare les candidats Lab+P et Lab+J à RGB, Lab6, au contrôle permuté et à la moyenne locale non conditionnée. Les quatre protocoles et les deux candidats forment une famille de 32 contrastes. `enriched.csv` contient les 16 contrastes d’enrichissement de RGB. Les IC ajustés dans chaque famille proviennent de 100 000 tirages appariés.
- `complementarity.csv` mesure l’apport du bloc chromatique aux seules ondelettes achromatiques A et à A+W. Les intervalles y sont ponctuels. `A` désigne les 945 coordonnées achromatiques. `AW` ajoute les 504 coordonnées conjointes achromatiques et chromatiques. `AW_RGB_J` partage le poids du troisième bloc à parts égales entre l’histogramme RGB et J. `only_` évalue le bloc couleur isolé. `__native` conserve les coefficients des termes survivants lors d’une ablation. Le tableau principal égalise les diagonales moyennes sur les données d’entraînement avant de comparer des nombres de blocs différents.
- `illumination.csv`, `matched_illumination.csv` et `lighting_pairs.csv` détaillent les éclairages KTH, le contrôle d’effectif et le test d’appariement frontal/ambiant à spécimen, pose et échelle inchangés. KTH comprend quatre spécimens par matériau. Le protocole ambiant réutilise KTH.
- `ALOT_scores.csv`, `ALOT_contrasts.csv`, `ALOT_decisions.csv` et `ALOT_splits.json` couvrent le suivi dans une autre acquisition. Les descripteurs mesurés sont dans `data/ALOT_inputs.npz`. Les six contrastes programmés sont ajustés ensemble. La même identité de texture est observée sous deux températures ou différentes incidences. Aucun transfert à une texture physique nouvelle n’est testé ici.
- `weight_control.csv` et `weight_control_decisions.npz` contiennent la sensibilité post-hoc. Dans le troisième bloc, le noyau comparateur utilise ½·similarité_RGB + ½·1. `RGB_plus_J` y remplace la constante par la similarité de J. Les bornes couvrent une famille de six contrastes.
- `CONTRASTE_NOYAU.md` définit une seconde analyse post-hoc, directement liée au manuscrit. `kernel_contrast.csv` donne l’effet d’ajouter les produits entre blocs au noyau à carrés, pour RGB et pour RGB + J, puis leur différence appariée avec et sans contrôle d’échelle. Une famille de 12 intervalles couvre les six protocoles et les deux réglages d’échelle. Les fichiers `*_interaction.npz` donnent les décisions correspondantes.
- Pour DTD, les dix tests officiels sont conservés et les sources identiques à une image test sont écartées du pool d’entraînement du pli. Les répétitions d’une même photo restent groupées en inférence. Pour KTH, le test tient à part un spécimen par matériau pour chacun des quatre plis. Pour FMD → Lytro, les 1 000 images sources FMD sont ajustées ensemble et les 1 000 vues monoculaires Lytro des dix matériaux communs sont testées sans adaptation. L’identité physique au-delà des identifiants de capture Lytro n’est pas documentée. Pour ALOT, les 250 identités entières sont rééchantillonnées. Tous les intervalles conditionnent sur les classifieurs ajustés.

## Calcul depuis les photographies

Les fichiers `data/*_inputs.npz` conservent identités, indices de classe, partitions où elles s’appliquent, SHA-256 des pixels 224 × 224 et blocs de départ. Les fichiers `*_new.npz` ordonnent les cinq moments locaux par axe et échelle, avec les indices du tableau `edges[image, P_ou_J, statistique, coordonnée]` dans l’ordre P, J puis alignée, moyenne, permutée. Les 9 valeurs Lab sont ordonnées par canaux L*, a*, b*, avec moyenne, écart-type et racine cubique du troisième moment centré. L’extraction est sans masque ni augmentation. Environ 5,5 Go sont nécessaires aux quatre archives téléchargées, davantage pour les images décompressées et le calcul.

Obtenir les photos auprès des fournisseurs. Les quatre empreintes d’archives se trouvent dans `source_archives.sha256`.

- DTD r1.0.1, [Visual Geometry Group, Oxford](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz).
- KTH-TIPS2-b couleur, [KTH](https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar).
- FMD, [MIT](https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip).
- Vues Lytro de Wang et al., [UC San Diego](https://visualcomp.ucsd.edu/projects/LF/papers/ECCV16/dataset.html). Le script récupère une vue pour chacune des 1 000 captures retenues.
- ALOT, version couleur au quart de la résolution native, [Université d’Amsterdam](https://aloi.science.uva.nl/public_alot/tars/alot_png4.tar). Conditions et partitions de référence [Université de Lille](https://color.univ-lille.fr/datasets/alot).

Pour les pixels, installer également Pillow, la version CPU de PyTorch et les bibliothèques d’ondelettes. Le profil natif reproduit aussi image par image les deux noyaux RGB du tableau 2 du manuscrit sur DTD et KTH. Notre contrôle d’échelle secondaire conserve ici le noyau complet et aligne le noyau à carrés sur celui-ci. Le tableau 2 de l’article prend l’autre noyau comme référence. Les valeurs des deux sensibilités dépendent de ce choix quand λ est fixé. Ces exemples utilisent un shell POSIX et un compilateur C++17.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-pixels.txt
mkdir -p inputs work measured
cp data/transfer_design.npz measured/
g++ -std=c++17 -O3 -shared -fPIC code/descriptor.cpp -o work/descriptor.so
python code/check_definition.py work/descriptor.so --output work/definition_checks.json
python code/collision.py work/descriptor.so work/collisions.csv
# Placer les quatre archives sous les noms indiqués dans source_archives.sha256.
(cd inputs && sha256sum -c ../source_archives.sha256)
python code/unpack_photos.py DTD inputs/dtd.tar.gz data inputs/DTD
python code/unpack_photos.py KTH inputs/kth.tar data inputs/KTH
python code/unpack_photos.py FMD inputs/FMD.zip data inputs/FMD
python code/fetch_target.py inputs/Lytro
python code/alot.py unpack inputs/alot_png4.tar inputs/ALOT
for corpus in DTD KTH FMD Lytro; do
  python code/extract_native.py "$corpus" "inputs/$corpus" data measured "work/${corpus}_native.json" --threads 2
  python code/extract.py "$corpus" "inputs/$corpus" measured work/descriptor.so
done
python code/alot.py extract inputs/ALOT work/descriptor.so measured/ALOT_inputs.npz --threads 2
python replay.py --data measured --threads 2 --log replay_depuis_pixels.json
```

`extract_native.py` utilise le lot de huit images et les noyaux CPU AVX2 du dossier rapporteurs. Il recalcule A, W et RGB, compare les coordonnées archivées et écrit le résultat dans le répertoire de mesure. `extract.py` vérifie le SHA de chaque image redimensionnée avant d’évaluer P, J, Lab et leurs témoins. La carte RGB native regroupe quatre niveaux par canal et prend la racine des 64 fréquences. `check_definition.py` confronte l’implémentation C++ à un calcul Python direct sur des données synthétiques. `collision.py` produit seize paires avec le même multiensemble de pixels RGB en couleur. Les photographies de tiers sont accessibles par les fournisseurs.

Le code du classifieur et des incertitudes se trouve dans `code/fit.py`, `code/stats.py` et `code/alot.py`. Les tests d’intégrité de nos exécutions se trouvent sous `results/*verified.json` et `results/*native_pixels.json`. Le rapport sépare les observations exploratoires, le suivi ALOT et le contrôle post-hoc de pondération.
