
from flare_dataset import get_prior12_span12
from basic_lstm import BasicLSTMModel
from train_basic_lstm import TrainLSTM


data_root = "/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARData"
model_dir = "/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Tensorboard/BasicLSTM"

def main():

	dataset = get_prior12_span12(data_root=data_root)
	lstm = BasicLSTMModel()
	train_lstm = TrainLSTM(lstm, dataset, model_dir=model_dir)
	train_lstm.train()


if __name__ == "__main__":
	main()