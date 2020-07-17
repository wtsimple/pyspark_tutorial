"""In this example we proof what we have developed, it also
allows us to make some decisions about what we'll do next

Unlike tests which are forced to be very fast,
examples could handle more realistic data"""
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from settings import ADULT_TRAIN_DATA_PATH, ADULT_COLUMN_NAMES, ADULT_TEST_DATA_PATH

if __name__ == '__main__':
    # Load data
    loader = DataLoader()
    train_df = loader.load_relative(path=ADULT_TRAIN_DATA_PATH, columns=ADULT_COLUMN_NAMES)
    test_df = loader.load_relative(path=ADULT_TEST_DATA_PATH, columns=ADULT_COLUMN_NAMES)

    # Explore data
    prep = DataPreprocessor(train_df, test_df)
    print("FACTOR COLUMNS", "#" * 40)
    prep.print_exploration(prep.explore_factors())
    print("NUMERIC COLUMNS", "#" * 40)
    prep.print_exploration(prep.explore_numeric_columns())
