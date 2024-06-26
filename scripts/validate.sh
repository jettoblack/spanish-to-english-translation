#!/bin/bash

cat $1 \
    | sed 's/\@\@ //g' \
    | ../tools/moses-scripts/scripts/recaser/detruecase.perl \
    | ../tools/moses-scripts/scripts/tokenizer/detokenizer.perl -l en \
    | ../tools/moses-scripts/scripts/generic/multi-bleu-detok.perl data/corpus-dev-test.en \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
