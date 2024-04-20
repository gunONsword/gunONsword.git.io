algos_graphormer.pyx, collating_graphormer.py, modeling_graphormer.py は
~/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/models/graphormer/
にあるoriginalと交換する。
/home/yasudak/data/Maillard/raw/に dataset_230720.pickle を置く。
/home/yasudak/data/Maillard/processed/に maillard_v0.pickle を置く。

conda activate huggingface
python graphormer_v1.py
を実行するとまず dataset_230720.pickle を読み、huggingface DatasetDict型にして /home/yasudak/data/Maillard/raw/train/, validation/, test/に保存する(graphormer_v1.py line 39)。
次に これらを読み preprocess_item を実行し、/home/yasudak/data/Maillard/processed/train/, validation/, test/に保存する(line 77)。
graphormer modelを新たに作り(line 130) 学習する(lines 183-193)。学習dataをbatchにまとめる時にGraphormerDataCollator()が呼ばれる。
学習したweightを保存する(line 207)。

data preprocessは2回目以後は省略できる(prep_data = False, line 38)。
学習済みweightでmodelを初期化するには line 131 を使う。

