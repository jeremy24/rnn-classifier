#!/usr/bin/fish

#source /home/ipp/virt_envs/tensorflow3_xla/bin/activate.fish


# set the lib path to enable tracing for cuda
set -g -x	LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# set the path to use tcmalloc instead
set -g -x LD_PRELOAD /usr/lib/libtcmalloc.so


set -g -x  CUDA_VISIBLE_DEVICES "0"


if	count $argv > /dev/null

	set dir $argv[1]
	set save_dir $dir"/save"

	set delete_results 1
	set n 500
	set delete_letters 1


	set concat_states 1

	if [ $concat_states = 1 ]
	    echo \n"Setting MODEL_USE_RNN_CONCAT = 1"\n
	    set -g -x  MODEL_USE_RNN_CONCAT 1
	end


	for arg in $argv
		switch "$arg"
			case --no_delete
				set delete_results 0
				set delete_letters 0
			end
	end

	if [ $delete_results = 1 ]
		echo "Removing results.."
		rm -f "./results/*"
	end

	if [ $delete_letters = 1 ]
		echo "Removing letters.."
		rm -f "./letters/*"
	end

	echo \n"Launching model..." \n
	python ./test.py --save_dir=$save_dir -n=$n

else
	echo "Please provide the models directory"
end
