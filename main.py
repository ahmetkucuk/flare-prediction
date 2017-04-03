
import sys
from flare_prediction import run

if __name__ == "__main__":
	run(sys.argv[1:])

	#run(["local", "min_max", "norm_min_max"])
	#run(["local", "z_score", "z_score"])
	#run(["local", "zero_center", "zero_center"])
