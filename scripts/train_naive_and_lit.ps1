foreach ($year in 1997..2023) {
    foreach ($target in 0..2) {
        python main_train.py --year $year --target $target --model-type lit naive
    }
}
