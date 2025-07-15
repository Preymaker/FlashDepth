config_path=""
output_dir=""
input_file=""

display_help() {
	echo "Usage: $0 [OPTIONS]"
	echo
	echo "Options:"
	echo "  -c, --config-path <path>   Path to the configuration directory."
	echo "  -o, --output-dir <dir>     Directory where output will be saved."
	echo "  -i, --input <file>         Input file to process."
	echo "  -h, --help                 Show this help message and exit."
	echo
	echo "Example:"
	echo "  $0 -c ./config.json -i input.txt -o ./out"
	echo
	exit 1
}

while [ "$1" != "" ]; do
	case $1 in
	--config-path | -c)
		shift
		config_path="$1"
		;;
	--output-dir | -o)
		shift
		output_dir="$1"
		;;
	--input | -i)
		shift
		input_file="$1"
		;;
	--help | -h)
		display_help
		;;
	*)
		echo "Error: Unknown option '$1'."
		display_help
		;;
	esac
	shift
done

# Validate paths and get full paths
if [ ! -d "$config_path" ]; then
	echo "Error: config_path '$config_path' is not a valid directory."
	exit 1
else
	config_path="$(cd "$config_path" && pwd)"
fi

if [ ! -d "$output_dir" ]; then
	echo "Error: output_dir '$output_dir' is not a valid directory."
	exit 1
else
	output_dir="$(cd "$output_dir" && pwd)"
fi

if [ ! -f "$input_file" ]; then
	echo "Error: input_file '$input_file' is not a valid file."
	exit 1
else
	input_file="$(cd "$(dirname "$input_file")" && pwd)/$(basename "$input_file")"
fi

# Extract the input file name
input_filename="$(basename "$input_file")"

echo "$config_path"
echo "$output_dir"
echo "$input_file"
echo "$input_filename"

# Check if XDG_RUNTIME_DIR is set and valid
if [ -z "$XDG_RUNTIME_DIR" ] || [ ! -d "$XDG_RUNTIME_DIR" ] || [ "$(stat -c %u "$XDG_RUNTIME_DIR" 2>/dev/null)" != "$(id -u)" ]; then
	echo "Warning: XDG_RUNTIME_DIR is unset or not owned by current user. Falling back to /tmp/xdg-$(id -u)"
	export XDG_RUNTIME_DIR="/tmp/xdg-$(id -u)"
	mkdir -p "$XDG_RUNTIME_DIR"
	chmod 700 "$XDG_RUNTIME_DIR"
fi

# TODO: I need to fix this so it doesn't use `sudo`
sudo podman run --rm \
	--device nvidia.com/gpu=all --security-opt=label=disable \
	-e TORCHINDUCTOR_DISABLE_BF16=1 \
	-v "$config_path":/app/configs \
	-v "$input_file":/app/examples/"$input_filename" \
	-v "$output_dir":/app/output \
	flashdepth \
	--config-path configs/flashdepth \
	inference=true \
	eval.random_input=examples/"$input_filename" \
	eval.outfolder=output
