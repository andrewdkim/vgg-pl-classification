from model import Model
import preprocess


if __name__ == "__main__":
    train_batches, test_batches = preprocess.process_images()

    model = Model()
    model.run(train_batches, test_batches)