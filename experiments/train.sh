#./[experiments/]train.sh <experiment_dir/experiment-file-w/o-extensions> [delay]
#./[experiments/]train.sh <experiment_dir> [delay]                                # using __main__.py
if [ -n "$2" ]; then
	echo "Waiting for $2!"
	sleep $2
fi

# get to the top-level directory
cd $(dirname "$0")/..

# use experiment.py or __main__.py
file="experiment.py"
[ ! -f "experiments/$path/$file" ] && file=""
[ "$file" ] && file=".$file" # module subscriptor

mkdir -p "experiments/$1" # pre-create the directory if neccessary

module=${1/\//.} # translate / into . for Python modules
# run training
# -u for unbuffered for Python, system("") makes flushing immediate
python -u -m "experiments.$module$file" | tee >(awk -F'\r' '{print $NF; system("")}' >"experiments/$1/log.txt")
