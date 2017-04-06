
import sys
from flare_prediction import run

if __name__ == "__main__":
	run(sys.argv[1:])

	#run(["local", "min_max", "norm_min_max"])
	#run(["local", "z_score", "norm_z_score"])
	#run(["local", "zero_center", "norm_zero_center"])
	#run(["local", "pca_whiten", "norm_pca_whiten"])
