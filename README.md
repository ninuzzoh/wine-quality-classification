# Wine Quality Classification — NLP con TF-IDF

Text classification per predire la qualità di un vino basandosi
esclusivamente sulla sua descrizione testuale, usando TF-IDF +
Multinomial Naive Bayes e Logistic Regression su 120.000 recensioni.

## Obiettivo

Addestrare un modello in grado di replicare il giudizio di un
sommelier professionista a partire dal testo della recensione.
Il punteggio numerico viene binarizzato sulla mediana (88 punti):
Buono (>= 88) vs Mediocre (< 88).

## Modelli confrontati

| Modello              | Accuracy | Precision | Recall | F1    |
|----------------------|----------|-----------|--------|-------|
| Multinomial Naive Bayes | 0.791 | 0.803  | 0.865  | 0.833 |
| Logistic Regression  | 0.831    | 0.846     | 0.879  | 0.862 |

La Logistic Regression supera il Naive Bayes di ~4 punti percentuali
catturando meglio le dipendenze tra parole (es. "not good").

## Caso di studio: vini siciliani

Il modello è stato testato su un sottoinsieme di 319 recensioni di
vini siciliani, ottenendo un'accuratezza dell'82.76% - in linea con
la performance globale.

## Pipeline

- Preprocessing: lowercasing, rimozione punteggiatura, stop words
- Vettorizzazione: TF-IDF con vocabolario limitato a 5.000 feature
  (unigram e bigram)
- Split: 80% training / 20% test

## Dataset

Wine Reviews - ~130.000 recensioni da Wine Magazine.
Scaricabile da Kaggle:
https://www.kaggle.com/datasets/zynicide/wine-reviews

## Esecuzione

```bash
pip install -r requirements.txt
jupyter notebook wine_quality_classification.ipynb
```

## Tecnologie

Python - scikit-learn - pandas - matplotlib - seaborn