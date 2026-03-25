# **Attention Is All You Need**

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

## **Abstract**

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

Work performed while at Google Brain.
Work performed while at Google Research.

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

## Wstęp
Sieci neuronowe rekurencyjne, w szczególności sieci z pamięcią krótkotrwałą i bramkowaną, zostały ugruntowane jako najlepsze podejścia w modelowaniu sekwencji i transdukcji, takich jak modelowanie języka i tłumaczenie maszynowe. Wiele wysiłków kontynuowało rozwijanie modeli językowych i architektur kodera-dekodera.

Modele rekurencyjne zwykle czynią obliczenia wzdłuż pozycji symboli wejściowych i wyjściowych sekwencji. Wyrównując pozycje do kroków w czasie obliczeń, generują sekwencję stanów ukrytych jako funkcję poprzedniego stanu ukrytego i wejścia dla danej pozycji. Ta wewnętrznie sekwencyjna natura uniemożliwia równoległość w ramach przykładów szkoleniowych, co staje się krytyczne przy dłuższych długościach sekwencji, ponieważ ograniczenia pamięci ograniczają grupowanie przykładów.

Mechanizmy uwagi stały się integralną częścią modeli sekwencyjnych i transdukcyjnych w różnych zadaniach, umożliwiając modelowanie zależności bez względu na ich odległość w sekwencjach wejściowych i wyjściowych. W niniejszej pracy proponujemy model Transformer, który odrzuca rekurencję i opiera się wyłącznie na mechanizmie uwagi, aby narysować globalne zależności między wejściem a wyjściem.

## Tło
Celem redukcji obliczeń sekwencyjnych jest również podstawą rozszerzonej sieci neuronowej GPU, ByteNet i ConvS2S, które wszystkie używają sieci neuronowych konwolucyjnych jako podstawowego bloku budulcowego, obliczając reprezentacje ukryte równolegle dla wszystkich pozycji wejściowych i wyjściowych. W tych modelach liczba operacji wymaganych do powiązania sygnałów z dwóch dowolnych pozycji wejściowych lub wyjściowych rośnie wraz z odległością między pozycjami, liniowo dla ConvS2S i logarytmicznie dla ByteNet. To sprawia, że trudniej jest nauczyć zależności między odległymi pozycjami. W modelu Transformer liczba ta jest zmniejszona do stałej liczby operacji, choć kosztem zmniejszonej skutecznej rozdzielczości z powodu uśredniania uwagowych pozycji, efekt ten jest przeciwdziałany przez uwagę wielogłową, jak opisano w sekcji 3.2.

Uwaga własna, czasem nazywana uwagą wewnątrzsekwencyjną, jest mechanizmem uwagi, który odnosi różne pozycje w sekwencji, aby obliczyć reprezentację sekwencji. Uwaga własna została z powodzeniem użyta w różnych zadaniach, w tym w czytaniu ze zrozumieniem, streszczeniu abstrakcyjnym, implikacji tekstowej i nauce reprezentacji zdaniowych niezależnych od zadania.

Sieci pamięciowe końca do końca opierają się na mechanizmie uwagi rekurencyjnej zamiast sekwencyjnej rekurencji i wykazały dobre wyniki w prostych zadaniach odpowiedzi na pytania językowe i modelowaniu języka.

Niniejsza praca opisuje model Transformer, motywuje uwagę własną i omawia jej zalety w porównaniu z modelami takimi jak [17, 18] i [9].

## Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations **z** = (z1, ..., zn). Given **z**, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

### Encoder and Decoder Stacks
**Encoder:** The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512.

**Decoder:** The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

### Attention
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

## 3.2.1 Scaled Dot-Product Attention
We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension _dk_, and values of dimension _dv_. We compute the dot products of the query with all keys, divide each by _√dk_, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix _Q_. The keys and values are also packed together into matrices _K_ and _V_. We compute the matrix of outputs as:

The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of _1/√dk_. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of _dk_ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of _dk_. We suspect that for large values of _dk_, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by _1/√dk_.

## 3.2.2 Multi-Head Attention
Instead of performing a single attention function with _d_ model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values _h_ times with different, learned linear projections to _dk_, _dk_ and _dv_ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding _dv_-dimensional output values. These are concatenated and once again projected, resulting in the final values.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

MultiHead(_Q, K, V_) = Concat(head1, ..., headh) _W_O_

Where the projections are parameter matrices _W_i_Q_ ∈ R _d_model_ × _d_k_, _W_i_K_ ∈ R _d_model_ × _d_k_, _W_i_V_ ∈ R _d_model_ × _d_v_ and _W_O_ ∈ R _h_d_v_ × _d_model_.

In this work we employ _h_ = 8 parallel attention layers, or heads. For each of these we use _dk_ = _dv_ = _d_model_ / _h_ = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

## 3.2.3 Zastosowania uwagi w naszym modelu

Transformer wykorzystuje uwagę wielogłową na trzy różne sposoby:

- W warstwach "uwagi dekodera-encoder" zapytania pochodzą z poprzedniej warstwy dekodera, a pamięć kluczy i wartości pochodzą z wyjścia encoder. Pozwala to każdej pozycji w dekoderze na uwagę nad wszystkimi pozycjami w sekwencji wejściowej. To naśladuje typowe mechanizmy uwagi w modelach sekwencja-do-sekwencji, takich jak [38, 2, 9].

- Encoder zawiera warstwy samouwagi. W warstwie samouwagi wszystkie klucze, wartości i zapytania pochodzą z tego samego miejsca, w tym przypadku, z wyjścia poprzedniej warstwy w encodera. Każda pozycja w encodera może zwrócić uwagę na wszystkie pozycje w poprzedniej warstwie encodera.

- Podobnie, warstwy samouwagi w dekoderze pozwalają każdej pozycji w dekoderze na uwagę nad wszystkimi pozycjami w dekoderze aż do i włącznie z tą pozycją. Musimy zapobiec przepływowi informacji w lewo w dekoderze, aby zachować właściwość autoregresyjną. Implementujemy to wewnątrz skalowanej uwagi punktowej, maskując (ustawiając na -∞) wszystkie wartości wejściowe softmax, które odpowiadają nielegalnym połączeniom.

## 3.3 Sieci neuronowe feed-forward w każdej pozycji

Oprócz warstw uwagi, każda z warstw w naszym encodera i dekoderze zawiera w pełni połączoną sieć neuronową, która jest stosowana do każdej pozycji oddzielnie i identycznie. Składa się z dwóch transformacji liniowych z aktywacją ReLU pomiędzy nimi.

FFN(x) = max(0, xW1 + b1)W2 + b2

Podczas gdy transformacje liniowe są takie same w różnych pozycjach, używają one różnych parametrów z warstwy do warstwy. Innym sposobem opisu tego jest jako dwie konwolucje z rozmiarem jądra 1. Wymiarowość wejścia i wyjścia to d_model = 512, a warstwa wewnętrzna ma wymiarowość dff = 2048.

## 3.4 Wbudowania i softmax

Podobnie jak w innych modelach transdukcji sekwencji, używamy nauczonych wbudowań, aby przekonwertować tokeny wejściowe i wyjściowe na wektory o wymiarze d_model. Używamy również standardowej nauczanej transformacji liniowej i funkcji softmax, aby przekonwertować wyjście dekodera na przewidywane prawdopodobieństwa następnego tokenu. W naszym modelu, dzielimy ten sam macierz wag pomiędzy dwie warstwy wbudowań i transformację liniową przed softmax, podobnie jak w [30]. W warstwach wbudowań, mnożymy te wagi przez √d_model.

Tabela 1: Maksymalne długości ścieżek, złożoność warstwy i minimalna liczba operacji sekwencyjnych dla różnych typów warstw. n jest długością sekwencji, d jest wymiarowością reprezentacji, k jest rozmiarem jądra konwolucji, a r jest rozmiarem sąsiedztwa w ograniczonej samouwadze.

|Typ warstwy|Złożoność na warstwę|Operacje sekwencyjne|Maksymalna długość ścieżki|
|---|---|---|---|
|Samouwaga|O(n^2 \* d)|O(1)|O(1)|
|Rekurencyjna|O(n \* d^2)|O(n)|O(n)|
|Konwolucyjna|O(k \* n \* d^2)|O(1)|O(log_k(n))|
|Samouwaga (ograniczona)|O(r \* n \* d)|O(1)|O(n/r)|

## 3.5 Positional Encoding
Od naszego modelu nie zawiera rekurencji i konwolucji, aby model mógł wykorzystywać kolejność sekwencji, musimy wstrzyknąć pewne informacje o względnej lub bezwzględnej pozycji tokenów w sekwencji. W tym celu dodajemy "kodowanie pozycyjne" do wejściowych wbudowań na dole stosu encodera i dekodera. Kodowanie pozycyjne ma tę samą wymiar _d_ modelu, co wbudowania, tak aby mogły być one sumowane. Istnieje wiele wyborów kodowania pozycyjnego, nauczonych i stałych [9].

W tej pracy używamy funkcji sinusoidalnych i cosinusoidalnych o różnych częstotliwościach:
gdzie _pos_ jest pozycją, a _i_ jest wymiarem. Innymi słowy, każdy wymiar kodowania pozycyjnego odpowiada sinusoidzie. Długości fal tworzą postęp geometryczny od 2 _π_ do 10000 _·_ 2 _π_. Wybraliśmy tę funkcję, ponieważ przypuszczaliśmy, że pozwoli to modelowi łatwo nauczyć się zwracać uwagę na względne pozycje, ponieważ dla każdego stałego przesunięcia _k_, _PEpos_ + _k_ może być reprezentowane jako funkcja liniowa _PEpos_.

Również eksperymentowaliśmy z użyciem nauczonych wbudowań pozycyjnych [9] zamiast, i stwierdziliśmy, że obie wersje dają niemal identyczne wyniki (patrz Tabela 3 wiersz (E)). Wybraliśmy wersję sinusoidalną, ponieważ może pozwolić modelowi ekstrapolować do długości sekwencji dłuższych niż te spotykane podczas treningu.

## Why Self-Attention

W tej sekcji porównujemy różne aspekty warstw self-attention z warstwami rekurencyjnymi i konwolucyjnymi, które są powszechnie używane do mapowania jednej sekwencji symboli o zmiennej długości (_x_1, ..., _x_n) na inną sekwencję tej samej długości (_z_1, ..., _z_n), gdzie _x_i, z_i ∈ R^d_, takich jak warstwa ukryta w typowym kodery lub dekodery sekwencji transdukcji. Motywacją naszego użycia self-attention są trzy pożądane cechy.

Pierwszą jest całkowita złożoność obliczeniowa na warstwę. Drugą jest ilość obliczeń, które mogą być równolegle przetwarzane, mierzone przez minimalną liczbę operacji sekwencyjnych wymaganych. Trzecią jest długość ścieżki między dalekimi zależnościami w sieci. Nauka dalekimi zależnościami jest kluczowym wyzwaniem w wielu zadaniach transdukcji sekwencji. Jednym z kluczowych czynników wpływających na zdolność do nauki takich zależności jest długość ścieżek, którymi sygnały muszą się poruszać w sieci. Im krótsze są te ścieżki między dowolnymi pozycjami wejściowymi i wyjściowymi, tym łatwiej jest nauczyć się dalekimi zależnościami [12]. Stąd również porównujemy maksymalną długość ścieżki między dowolnymi dwiema pozycjami wejściowymi i wyjściowymi w sieciach zbudowanych z różnych typów warstw.

Jak zauważono w Tabeli 1, warstwa self-attention łączy wszystkie pozycje z stałą liczbą operacji wykonywanych sekwencyjnie, podczas gdy warstwa rekurencyjna wymaga _O(n)_ operacji sekwencyjnych. Pod względem złożoności obliczeniowej, warstwy self-attention są szybsze niż warstwy rekurencyjne, gdy długość sekwencji _n_ jest mniejsza niż wymiarowość reprezentacji _d_, co jest najczęstszym przypadkiem w przypadku reprezentacji zdań używanych przez modele stanu sztuki w tłumaczeniach maszynowych, takich jak reprezentacje word-piece [38] i byte-pair [31]. Aby poprawić wydajność obliczeniową dla zadań z bardzo długimi sekwencjami, self-attention można ograniczyć do rozważania tylko sąsiedztwa o rozmiarze _r_ w sekwencji wejściowej, skupionej wokół odpowiedniej pozycji wyjściowej. To zwiększyłoby maksymalną długość ścieżki do _O(n/r)_. Planujemy zbadanie tego podejścia w przyszłej pracy.

Jedna warstwa konwolucyjna o szerokości jądra _k < n_ nie łączy wszystkich par pozycji wejściowych i wyjściowych. Aby to zrobić, wymaga się stosu _O(n/k)_ warstw konwolucyjnych w przypadku jąder ciągłych lub _O(log_k(n))_ w przypadku rozcieńczonych konwolucji [18], co zwiększa długość najdłuższych ścieżek między dowolnymi dwiema pozycjami w sieci. Warstwy konwolucyjne są geralnie droższe niż warstwy rekurencyjne, o czynnik _k_. Separable konwolucje [6] zmniejszają jednak znacznie złożoność, do _O(k · n · d + n · d^2)_. Nawet z _k = n_, złożoność separable konwolucji jest równa połączeniu warstwy self-attention i warstwy feed-forward punktowej, co jest podejściem, które stosujemy w naszym modelu.

Jako dodatkowy benefit, self-attention może prowadzić do bardziej interpretowalnych modeli. Inspekcja rozkładów uwagi z naszych modeli i przedstawienie oraz dyskusja przykładów w dodatku. Nie tylko poszczególne głowy uwagi wyraźnie uczą się wykonywać różne zadania, wiele z nich wydaje się wykazywać zachowanie związane ze strukturą składniową i semantyczną zdań.

## Trening

Ta sekcja opisuje reżim treningu dla naszych modeli.

## Training
This section describes the training regime for our models.

## Training Data and Batching
We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

## Hardware and Schedule
We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models, step time was 1.0 seconds. The big models were trained for 300,000 steps.

## Optimizer
We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ϵ = 10^(-9). We varied the learning rate over the course of training, according to the formula. This corresponds to increasing the learning rate linearly for the first warmup steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup steps = 4000.

## Regularization
We employ three types of regularization during training.

The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

| Model | BLEU EN-DE EN-FR | Training Cost (FLOPs) |
| --- | --- | --- |
| ByteNet | 23.75 39.2 | 1.0 · 10^20 |
| Deep-Att + PosUnk | 24.6 39.92 | 2.3 · 10^19 |
| GNMT + RL | 25.16 40.46 | 1.4 · 10^20 |
| ConvS2S | 26.03 40.56 | 9.6 · 10^18 |
| MoE |  | 1.5 · 10^20 |
| Deep-Att + PosUnk Ensemble | 40.4 26.30 | 8.0 · 10^20 |
| GNMT + RL Ensemble | 26.30 41.16 | 1.8 · 10^20 |
| ConvS2S Ensemble | 41.16 26.36 | 1.1 · 10^21 |
| Transformer (base model) | 27.3 38.1 | 3.3 · 10^18 |
| Transformer (big) | 28.4 41.8 | 2.3 · 10^19 |

### Residual Dropout
We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1.

### Label Smoothing
During training, we employed label smoothing of value ϵls = 0.1. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

## Machine Translation

Na zadaniu tłumaczenia z języka angielskiego na niemiecki WMT 2014, model transformatora o dużej skali (Transformer (big) w Tabeli 2) przewyższa najlepsze wcześniej opublikowane modele (w tym zespoły) o ponad 2,0 BLEU, ustanawiając nowy stan techniki wynik BLEU na poziomie 28,4. Konfiguracja tego modelu jest wymieniona w dolnej linii Tabeli 3. Trening trwał 3,5 dni na 8 procesorach graficznych P100. Nawet nasz model podstawowy przewyższa wszystkie wcześniej opublikowane modele i zespoły, przy ułamku kosztu treningu jakiegokolwiek z konkurencyjnych modeli.

Na zadaniu tłumaczenia z języka angielskiego na francuski WMT 2014, nasz duży model osiąga wynik BLEU na poziomie 41,0, przewyższając wszystkie wcześniej opublikowane pojedyncze modele, przy mniej niż 1/4 kosztu treningu poprzedniego modelu stanu techniki. Model Transformer (big) przeszkolony dla języka angielskiego na francuski używał współczynnika dropout _Pdrop_ = 0,1, zamiast 0,3.

Dla modeli podstawowych, użyliśmy pojedynczego modelu uzyskanego przez uśrednienie ostatnich 5 punktów kontrolnych, które były zapisywane w odstępach 10-minutowych. Dla dużych modeli, uśredniliśmy ostatnie 20 punktów kontrolnych. Użyliśmy wyszukiwania wiązki z rozmiarem wiązki 4 i karą długości _α_ = 0,6 [38]. Te hiperparametry zostały wybrane po eksperymentach na zbiorze danych rozwojowych. Ustawiliśmy maksymalną długość wyjścia podczas inferencji na długość wejścia + 50, ale zakończyliśmy wcześniej, gdy było to możliwe [38].

Tabela 2 podsumowuje nasze wyniki i porównuje naszą jakość tłumaczenia i koszty treningu z innymi architekturami modeli z literatury. Szacujemy liczbę operacji zmiennoprzecinkowych używanych do treningu modelu, mnożąc czas treningu, liczbę procesorów graficznych używanych i szacowaną wydajność zmiennoprzecinkową każdego procesora graficznego [5].

## Model Variations
To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

In Table 3 rows (B), we observe that reducing the attention key size _dk_ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings, and observe nearly identical results to the base model.

### Tabela 3: Wariacje architektury Transformer
| _N_ | _d_model_ | _d_ff_ | _h_ | _dk_ | _dv_ | _Pdrop_ | _ϵls_ | train steps | PPL | BLEU | params (dev) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) | 1 | 512 | 512 | 4 | 128 | 128 | 16 | 32 | 32 | 5.29 | 24.9 | 5.00 |
| (B) | 16 | 32 |  |  |  |  |  |  |  | 5.16 | 25.1 | 58 |
| (C) | 2 | 4 | 8 | 256 | 32 | 32 | 1024 | 128 | 128 | 6.11 | 23.7 | 36 |
| (D) | 0.0 | 0.2 | 0.0 | 0.2 |  |  |  |  |  | 5.77 | 24.6 | 4.95 |
| (E) |  |  |  |  |  |  |  |  |  | 4.92 | 25.7 |  |
| big | 6 | 1024 | 4096 | 16 |  | 0.3 | 300K |  |  | **4.33** | **26.4** | 213 |

## English Constituency Parsing
To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes.

We trained a 4-layer transformer with _dmodel_ = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank, about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual, learning rates and beam size on the development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + 300. We used a beam size of 21 and _α_ = 0.3 for both WSJ only and the semi-supervised setting.

Our results show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar. In contrast to RNN sequence-to-sequence models, the Transformer outperforms the BerkeleyParser even when training only on the WSJ training set of 40K sentences.

## Conclusion
In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goal of ours.

The code we used to train and evaluate our models is available at `https://github.com/tensorflow/tensor2tensor`.

### Wyniki
| Parser | Training | WSJ 23 F1 |
| --- | --- | --- |
| Vinyals & Kaiser el al. (2014) | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) | WSJ only, discriminative | 91.7 |
| Transformer (4 layers) | WSJ only, discriminative | 91.3 |
| Zhu et al. (2013) | semi-supervised | 91.3 |
| Huang & Harper (2009) | semi-supervised | 91.3 |
| McClosky et al. (2006) | semi-supervised | 92.1 |
| Vinyals & Kaiser el al. (2014) | semi-supervised | 92.1 |
| Transformer (4 layers) | semi-supervised | 92.7 |
| Luong et al. (2015) | multi-task | 93.0 |
| Dyer et al. (2016) | generative | 93.3 |

### Podziękowania
We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. _arXiv preprint arXiv:1607.06450_ , 2016.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. _CoRR_ , abs/1409.0473, 2014.

Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. _CoRR_ , abs/1703.03906, 2017.

Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. _arXiv preprint arXiv:1601.06733_ , 2016.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. _CoRR_ , abs/1406.1078, 2014.

Francois Chollet. Xception: Deep learning with depthwise separable convolutions. _arXiv preprint arXiv:1610.02357_ , 2016.

Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. _CoRR_ , abs/1412.3555, 2014.

Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In _Proc. of NAACL_ , 2016.

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. _arXiv preprint arXiv:1705.03122v2_ , 2017.

Alex Graves. Generating sequences with recurrent neural networks. _arXiv preprint arXiv:1308.0850_ , 2013.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 770–778, 2016.

Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural computation_ , 9(8):1735–1780, 1997.

Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In _Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing_ , pages 832–841. ACL, August 2009.

Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. _arXiv preprint arXiv:1602.02410_ , 2016.

Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In _Advances in Neural Information Processing Systems, (NIPS)_ , 2016.

Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In _International Conference on Learning Representations (ICLR)_ , 2016.

Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. _arXiv preprint arXiv:1610.10099v2_ , 2017.

Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In _International Conference on Learning Representations_ , 2017.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _ICLR_ , 2015.

Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. _arXiv preprint arXiv:1703.10722_ , 2017.

Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. _arXiv preprint arXiv:1703.03130_ , 2017.

Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. _arXiv preprint arXiv:1511.06114_ , 2015.

Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. _arXiv preprint arXiv:1508.04025_ , 2015.

Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. _arXiv preprint arXiv:1508.04025_ , 2015.
Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. _Computational linguistics_ , 19(2):313–330, 1993.
David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In _Proceedings of the Human Language Technology Conference of the NAACL, Main Conference_ , pages 152–159. ACL, June 2006.
Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In _Empirical Methods in Natural Language Processing_ , 2016.
Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. _arXiv preprint arXiv:1705.04304_ , 2017.
Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In _Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL_ , pages 433–440. ACL, July 2006.
Ofir Press i Lior Wolf. Using the output embedding to improve language models. _arXiv preprint arXiv:1608.05859_ , 2016.
Rico Sennrich, Barry Haddow, i Alexandra Birch. Neural machine translation of rare words with subword units. _arXiv preprint arXiv:1508.07909_ , 2015.
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, i Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. _arXiv preprint arXiv:1701.06538_ , 2017.
Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, i Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. _Journal of Machine Learning Research_ , 15(1):1929–1958, 2014.
Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, i Rob Fergus. End-to-end memory networks. W C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, i R. Garnett, redaktorach, _Advances in Neural Information Processing Systems 28_ , pages 2440–2448. Curran Associates, Inc., 2015.
Ilya Sutskever, Oriol Vinyals, i Quoc VV Le. Sequence to sequence learning with neural networks. W _Advances in Neural Information Processing Systems_ , pages 3104–3112, 2014.
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, i Zbigniew Wojna. Rethinking the inception architecture for computer vision. _CoRR_ , abs/1512.00567, 2015.
Vinyals i Kaiser, Koo, Petrov, Sutskever, i Hinton. Grammar as a foreign language. W _Advances in Neural Information Processing Systems_ , 2015.
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. _arXiv preprint arXiv:1609.08144_ , 2016.
Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, i Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. _CoRR_ , abs/1606.04199, 2016.
Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, i Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. W _Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers)_ , pages 434–443. ACL, August 2013.

## **Attention Visualizations**

It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult.
It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult.

Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.

The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.
The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.
The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.
The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.

The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.
The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.
The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.
The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.