from dataset import COVISDataset
from model import Model
import preprocess
from torch.utils.data import DataLoader

if __name__ == "__main__":

    RBDataset = COVISDataset("data/RB")
    # train_batches_RB, test_batches_RB = preprocess.process_dataset(RBDataset)
    # model = Model()
    # model.run(train_batches_RB, test_batches_RB, "output/RB_Model")


    IIDataset = COVISDataset("data/II")
    # train_batches_II, test_batches_II = preprocess.process_dataset(IIDataset)
    # model = Model()
    # model.run(train_batches_II, test_batches_II, "output/II_Model")
    
    
    # model.create_vectors(RBDataset)
    

    # model = Model() 
    # model.run(train_batches_RB, test_batches_RB)